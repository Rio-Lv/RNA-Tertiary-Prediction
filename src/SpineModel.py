from __future__ import annotations

import os, pathlib
import random
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import List
from tools import *


EPS = 1e-8  # avoids 0‑division
MAX_DELTA = 0.01  # clip per‑step movement (Å)
N_ITER = 3000  # relax steps *after each point*


# ------------------------- hyper‑parameters ------------------------- #
SPINE_TRAIN_EPOCHS = 5000
SPINE_MODEL_LR = 0.001
SPINE_TRAIN_BATCH_SIZE = 64**2
SPINE_DATA_TRAIN_FRAC = 0.8
SPINE_WINDOW_SIZE = 5
SPINE_N_SEQUENCES = 10_000

DATASET_PATH = pathlib.Path("data/spine_dataset.pt")
MODEL_PATH = pathlib.Path("models/spine_model.pt")

# flip either flag to *True* before running to wipe the corresponding cache
RESET_DATA = False
RESET_MODEL = False
os.chdir(pathlib.Path(__file__).parent.resolve())
# ------------------------ cache management ------------------------- #


def _delete_path(p: pathlib.Path) -> None:
    if p.is_file():
        p.unlink()
        print(f"=== Deleted {p} ===")


# delete on demand
if RESET_DATA:
    _delete_path(DATASET_PATH)
if RESET_MODEL:
    _delete_path(MODEL_PATH)

# ----------------------------- helpers ----------------------------- #

def plot_distance_heatmap(
    distance_matrix: torch.Tensor, title: str = "Distance matrix"
) -> None:
    """Plot a square heat‑map for a (L×L) distance matrix."""
    plt.figure()
    plt.imshow(distance_matrix.cpu(), aspect="equal")
    plt.colorbar(label="Distance")
    plt.title(title)
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.tight_layout()
    plt.show()


def save_artifact(obj: Any, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_artifact(path: pathlib.Path):
    return torch.load(path, map_location="cpu") if path.is_file() else None


# ------------------------------------------------------------------- #
#  PyTorch ≥ 2.3 safe‑load note
# ------------------------------------------------------------------- #
# We cache the dataset as a *dict of tensors* to avoid unpickling globals;
# see https://pytorch.org/docs/stable/serialization.html for details.
# ------------------------------------------------------------------- #


def _reconstruct_ds(cache: Dict[str, torch.Tensor]) -> TensorDataset:
    return TensorDataset(cache["x"], cache["y"])


# --------------------------- core model ---------------------------- #


class SpineModel(nn.Module):
    """Simple MLP predicting a distance matrix from a one-hot encoding."""

    def __init__(
        self, sequence_size: int = SPINE_WINDOW_SIZE, lr: float = SPINE_MODEL_LR
    ):
        super().__init__()
        self.sequence_size = sequence_size

        self.model = nn.Sequential(
            nn.Linear(4 * sequence_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, sequence_size * sequence_size),
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.train_history: list[float] = []
        self.val_history: list[float] = []

        # Load weights if present so we *continue* training instead of starting over
        state = load_artifact(MODEL_PATH)
        if state is not None:
            self.model.load_state_dict(state)
            print(
                f"=== Loaded model weights from {MODEL_PATH}; continuing training ==="
            )

    # .................................................................
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x.view(x.size(0), -1)).view(
            -1, self.sequence_size, self.sequence_size
        )

    # ---------------------------- data ------------------------------ #

    @staticmethod
    def _build_tensor_dataset(n_sequences: int, sequence_size: int) -> TensorDataset:
        from Sequence import SequenceDataset  # local import

        seq_ds = SequenceDataset(n_sequences=n_sequences, sequence_size=sequence_size)
        xs, ys = zip(
            *((s.encoding, s.distance_matrix) for s in seq_ds.source_sequences)
        )
        x = torch.stack(xs).float()
        y = torch.stack(ys).float()
        mask = torch.isfinite(x.view(x.size(0), -1)).all(1) & torch.isfinite(
            y.view(y.size(0), -1)
        ).all(1)
        return TensorDataset(x[mask], y[mask])

    def get_dataloaders(
        self, batch_size: int = SPINE_TRAIN_BATCH_SIZE
    ) -> Tuple[DataLoader, DataLoader]:
        cache = load_artifact(DATASET_PATH)
        if cache is None:
            print("=== Generating dataset ===")
            ds = self._build_tensor_dataset(SPINE_N_SEQUENCES, self.sequence_size)
            save_artifact({"x": ds.tensors[0], "y": ds.tensors[1]}, DATASET_PATH)
            print(f"=== Saved dataset to {DATASET_PATH} ===")
        else:
            print(f"=== Loaded dataset from {DATASET_PATH} ===")
            ds = _reconstruct_ds(cache)
            # shuffle the dataset to avoid overfitting
            indices = torch.randperm(len(ds))
            ds = TensorDataset(ds.tensors[0][indices], ds.tensors[1][indices])
            print("=== Shuffled dataset ===")

        n_train = int(SPINE_DATA_TRAIN_FRAC * len(ds))
        n_val = len(ds) - n_train
        train_ds, val_ds = random_split(
            ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        )

    # --------------------------- train ------------------------------ #

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = SPINE_TRAIN_EPOCHS,
    ):
        for ep in range(epochs):
            # --- train ---
            self.train()
            tr_loss, tr_n = 0.0, 0
            for x, y in train_loader:
                self.optimizer.zero_grad()
                loss = self.loss_fn(self(x), y)
                loss.backward()
                self.optimizer.step()
                tr_loss += loss.item() * x.size(0)
                tr_n += x.size(0)
            self.train_history.append(tr_loss / tr_n)

            # --- val ---
            self.eval()
            v_loss, v_n = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    v_loss += self.loss_fn(self(x), y).item() * x.size(0)
                    v_n += x.size(0)
            self.val_history.append(v_loss / v_n)

            if ep % 100 == 0:
                print(
                    f"epoch {ep:>5}/{epochs}  train={self.train_history[-1]:.4f}  val={self.val_history[-1]:.4f}"
                )

        save_artifact(self.model.state_dict(), MODEL_PATH)
        print(f"=== Saved model weights to {MODEL_PATH} ===")

    # --------------------------- utils ----------------------------- #

    def plot_history(self):
        plt.figure()
        plt.plot(self.train_history, label="train")
        plt.plot(self.val_history, label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.ylim(0, 10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @torch.no_grad()  # no gradients anywhere in this function
    def construct_distance_matrix(self, seq_str: str) -> torch.Tensor:
        L = len(seq_str)
        W = SPINE_WINDOW_SIZE  # window length
        device = next(self.parameters()).device  # handle cpu / cuda transparently

        # running totals
        sum_matrix = torch.zeros((L, L), dtype=torch.float32, device=device)
        cnt_matrix = torch.zeros((L, L), dtype=torch.float32, device=device)

        # helper to turn a character into one‑hot
        vocab = {"A": 0, "C": 1, "G": 2, "U": 3}
        eye4 = torch.eye(4, dtype=torch.float32, device=device)

        for i in range(L - W + 1):
            # one‑hot encode window  (W, 4)
            idxs = [vocab.get(ch, -1) for ch in seq_str[i : i + W]]
            enc = torch.stack(
                [(eye4[j] if j >= 0 else torch.zeros(4, device=device)) for j in idxs]
            )

            # (1, 4, W) → model → (W, W)
            pred = self(enc.unsqueeze(0)).squeeze(0)  # (W, W)

            # accumulate
            r = slice(i, i + W)
            sum_matrix[r, r] += pred
            cnt_matrix[r, r] += 1.0

        # avoid division by zero (off‑diagonal never touched for extremely short seqs)
        mask = cnt_matrix > 0
        distance_matrix = torch.zeros_like(sum_matrix)
        distance_matrix[mask] = sum_matrix[mask] / cnt_matrix[mask]

        # optional: print nicely
        torch.set_printoptions(precision=4, sci_mode=False)

        return distance_matrix.cpu()  # return on CPU for convenience

    # -------------------------------------------------------------------- #
    #  main driver
    # -------------------------------------------------------------------- #
    
    def construct_spine_coords(
        self, seq_str: str, n_iter: int = N_ITER, max_delta: float = MAX_DELTA
    ) -> List[Vector]:
        coords = [Vector(
            x=random.uniform(-1, 1),
            y=random.uniform(-1, 1),
            z=random.uniform(-1, 1),
            ) for _ in range(len(seq_str))]
        distance_matrix = self.construct_distance_matrix(seq_str)
        coords, recording = adjust_coords(
            n_iter=n_iter,
            coords=coords,
            target_matrix=distance_matrix,
            max_delta=max_delta,
            iterations_per_residue=1,
            temperature=0.1,
            delta_drop_rate=0,
            max_index_diff=SPINE_WINDOW_SIZE
        )
        return coords
        

# --------------------------- script entry -------------------------- #

if __name__ == "__main__":

    # # 1. ===== TRAINING THE MODEL =====
    # torch.set_printoptions(precision=4, sci_mode=False)

    # model = SpineModel()
    # tr_loader, vl_loader = model.get_dataloaders()

    # # ── grab one validation sample ──────────────────────────────────
    # sample_x, target = vl_loader.dataset[0]          # (4, L) , (L, L)
    # sample_x = sample_x.unsqueeze(0)                 # -> (1, 4, L)

    # with torch.no_grad():
    #     out_before = model(sample_x).squeeze(0)      # (L, L)
    #     mse_before = nn.functional.mse_loss(out_before, target).item()

    # # ── train (continues from cached weights if any) ────────────────
    # model.fit(tr_loader, vl_loader)

    # with torch.no_grad():
    #     out_after  = model(sample_x).squeeze(0)
    #     mse_after  = nn.functional.mse_loss(out_after, target).item()

    # # ── nicely formatted report ─────────────────────────────────────
    # print(f"\nInput encoding shape : {sample_x.shape[1:]}")   # (4, L)
    # print(f"Target distance shape : {target.shape}")          # (L, L)

    # print(f"\n>>> TARGET")
    # print(target)

    # print(f"\n>>> BEFORE training — MSE vs. target = {mse_before:.4f}")
    # print(out_before)

    # print(f"\n>>> AFTER  training — MSE vs. target = {mse_after:.4f}")
    # print(out_after)

    # # absolute error matrices (optional, comment out if too verbose)
    # print("\n|before - target| :")
    # print((out_before - target).abs())

    # print("\n|after - target|  :")
    # print((out_after  - target).abs())

    # model.plot_history()

    # 2. ===== USING THE MODEL =====
    spine_model = SpineModel()
    # create a distance matrix for a random sequence
    distance_matrix = spine_model.construct_distance_matrix("ACGUAAAA")
    spine_coords = spine_model.construct_spine_coords("ACGUAACGCGUAUAUAUCACACACUCUCUGCGCGC")
    # plot the coordinates
    plot_coords_list([spine_coords], ["Spine Coordinates"])
    plot_distance_heatmap(distance_matrix)

