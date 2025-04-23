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
import torch.nn.functional as F
from SpineModel import SpineModel

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


EPS = 1e-8  # avoids 0‑division
MAX_DELTA = 0.1  # clip per‑step movement (Å)
SPINE_ITERATIONS_PER_RESIDUE = 20
ITERATIONS_TO_SETTLE = 1000
TEMPERATURE = 0.1
ACITVE_KEEP_RATE_SPINE = 1
ACITVE_KEEP_RATE_SETTLE = 0.1

MAX_INDEX_DIFF = 0


# ------------------------- hyper‑parameters ------------------------- #
TRAIN = False

# flip either flag to *True* before running to wipe the corresponding cache
RESET_DATA = False
RESET_MODEL = False

SPINE_TRAIN_EPOCHS = 5_000
SPINE_N_SEQUENCES = 20_000
SPINE_MODEL_LR = 0.001
SPINE_TRAIN_BATCH_SIZE = 128**2
SPINE_DATA_TRAIN_FRAC = 0.8

SPINE_WINDOW_SIZE = 24

DATASET_PATH = pathlib.Path("data/spine_dataset_big.pt")
MODEL_PATH = pathlib.Path("models/spine_model_big.pt")


os.chdir(pathlib.Path(__file__).parent.resolve())
# ------------------------ cache management ------------------------- #


def _delete_path(p: pathlib.Path) -> None:
    if p.is_file():
        p.unlink()
        print(f"=== Deleted {p} ===")


# ----------------------------- helpers ----------------------------- #


def save_artifact(obj: Any, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_artifact(path: pathlib.Path):
    return torch.load(path, map_location=device) if path.is_file() else None


# ------------------------------------------------------------------- #
#  PyTorch ≥ 2.3 safe‑load note
# ------------------------------------------------------------------- #
# We cache the dataset as a *dict of tensors* to avoid unpickling globals;
# see https://pytorch.org/docs/stable/serialization.html for details.
# ------------------------------------------------------------------- #


def _reconstruct_ds(cache: Dict[str, torch.Tensor]) -> TensorDataset:
    return TensorDataset(cache["x"], cache["y"])


# --------------------------- core model ---------------------------- #


class SpineModelBig(nn.Module):
    """Simple MLP predicting a distance matrix from a one-hot encoding."""

    def __init__(self, subset_len: int = SPINE_WINDOW_SIZE, lr: float = SPINE_MODEL_LR):
        """ 
        Model to predict distance matrix from one-hot encoding of a sequence.
        Input shape: (batch, subset_len, 4)
        Output shape: (batch, subset_len, subset_len)
        """
        
        super().__init__()
        self.lr = lr
        
        
        self.subset_len = subset_len
        self.val_history = []
        self.train_history = []

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,1), padding=(1,0))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,1), padding=(1,0))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,1), padding=(1,0))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,1), padding=(1,0))
        self.dropout = nn.Dropout(0.7)
        # new: kernel=(3,4) collapses the width=4
        self.conv5 = nn.Conv2d(64, subset_len, kernel_size=(3,4), padding=(1,0))

        self.loss_fn = nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=lr)

        # continue training if weights exist
        state = load_artifact(MODEL_PATH)
        if state is not None:
            self.load_state_dict(state)
            print(f"=== Loaded model weights from {MODEL_PATH} ===")

    def forward(self, x):
        # x: (batch, subset_len, 4)
        x = x.unsqueeze(1)             # → (batch, 1, subset_len, 4)
        x = F.relu(self.conv1(x))      # → (batch, 32, subset_len, 4)
        x = F.relu(self.conv2(x))      # → (batch, 64, subset_len, 4)
        x = F.relu(self.conv3(x))      # → (batch, 64, subset_len, 4)
        x = F.relu(self.conv4(x))      # → (batch, 64, subset_len, 4)
        x = self.dropout(x)
        x = self.conv5(x)              # → (batch, subset_len, subset_len, 1)
        return x.squeeze(3)            # → (batch, subset_len, subset_len)

    # ---------------------------- data ------------------------------ #

    @staticmethod
    def _build_tensor_dataset(n_sequences: int, subset_len: int) -> TensorDataset:
        from Sequence import SequenceDataset  # local import

        seq_ds = SequenceDataset(n_sequences=n_sequences, subset_len=subset_len)
        xs, ys = zip(
            *((s.encoding, s.distance_matrix) for s in seq_ds.subset_sequences)
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
            ds = self._build_tensor_dataset(SPINE_N_SEQUENCES, self.subset_len)
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
                x, y = x.to(device), y.to(device)
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
                    x, y = x.to(device), y.to(device)
                    v_loss += self.loss_fn(self(x), y).item() * x.size(0)
                    v_n += x.size(0)
            self.val_history.append(v_loss / v_n)

            if ep % 100 == 0:
                print(
                    f"epoch {ep:>5}/{epochs}  train={self.train_history[-1]:.4f}  val={self.val_history[-1]:.4f}"
                )
                save_artifact(
                    self.state_dict(), MODEL_PATH
                )  # save every 100 epochs

        print("train val history", self.train_history, self.val_history)

        save_artifact(self.state_dict(), MODEL_PATH)
        print(f"=== Saved model weights to {MODEL_PATH} ===")

    # --------------------------- utils ----------------------------- #

    def plot_history(self):
        plt.plot(self.train_history, label="train")
        plt.plot(self.val_history, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training history")
        plt.legend()
        plt.show()

    @torch.no_grad()
    def construct_distance_matrix(self, seq_str: str) -> torch.Tensor:
        count = torch.ones(len(seq_str), len(seq_str), device=device)
        distance_matrix = torch.zeros(len(seq_str), len(seq_str), device=device)

        last_index = len(seq_str) - self.subset_len + 1
        for i in range(last_index):
            min_index = i
            max_index = i + self.subset_len
            # slice the sequence and encode it
            seq_slice = seq_str[i : i + self.subset_len]
            seq_onehot: Tensor = encode_str(seq_slice).to(device)
            dist_mat: Tensor = self.forward(seq_onehot.unsqueeze(0)).squeeze(0)
            dist_mat = dist_mat.view(self.subset_len, self.subset_len)

            distance_matrix[min_index:max_index, min_index:max_index] += dist_mat

            count[min_index:max_index, min_index:max_index] += 1

        distance_matrix /= count
        # plot_distance_heatmap(
        #     distance_matrix, title=f"Distance matrix for {seq_str}"
        # )
        return distance_matrix 
       

    # -------------------------------------------------------------------- #
    #  main driver
    # -------------------------------------------------------------------- #

    def construct_spine_coords(
        self,
        seq_str: str,
        temperature: float = TEMPERATURE,
        iterations_per_residue: int = SPINE_ITERATIONS_PER_RESIDUE,
        iterations_to_settle: int = ITERATIONS_TO_SETTLE,
        max_delta: float = MAX_DELTA,
        active_keep_rate_spine: float = ACITVE_KEEP_RATE_SPINE,
        active_keep_rate_settle: float = ACITVE_KEEP_RATE_SETTLE,
        target_matrix: Tensor = None,
    ) -> Tuple[List[Vector], List[Tensor]]:
        """
        1. Put down first
        """

        coords: list[Vector] = []
        full_recording: list[Tensor] = []
        spine_distance_matrix = self.construct_distance_matrix(seq_str).to(device)
        # plot_distance_heatmap(spine_distance_matrix)
        distance_matrix = torch.zeros(len(seq_str), len(seq_str)).to(device)
        # Use Real Matrix but Replace Spine
        # Use Real Matrix but Replace Spine
        if target_matrix is not None:
            for i in range(len(seq_str)):
                for j in range(len(seq_str)):
                    if abs(i - j) < MAX_INDEX_DIFF:
                        distance_matrix[i, j] = target_matrix[i, j]
                        
        for i in range(len(seq_str)):
            for j in range(len(seq_str)):
                if abs(i - j) < SPINE_WINDOW_SIZE:
                    distance_matrix[i, j] = spine_distance_matrix[i, j]
        small_spine = SpineModel()
        small_spine.to(device)
        small_spine_seq = small_spine.subset_len
        small_spine_distance_matrix = small_spine.construct_distance_matrix(
            seq_str=seq_str
        ).to(device)
        for i in range(len(seq_str)):
            for j in range(len(seq_str)):
                if abs(i - j) < small_spine_seq:
                    distance_matrix[i, j] = small_spine_distance_matrix[i, j]
                        

        noise = 0.1
        for i in range(len(seq_str)):
            if i % 20 == 0:
                print(f"Nucleotide {i}/{len(seq_str)}")
            if i < SPINE_WINDOW_SIZE:
                coords.append(
                    Vector(
                        i*3.5+random.uniform(-noise, noise),
                        random.uniform(-noise, noise),
                        random.uniform(-noise, noise),
                    )
                )
            else:
                coord_1 = coords[-1].copy()
                coord_2 = coords[-2].copy()
                dx = coord_2.x - coord_1.x
                dy = coord_2.y - coord_1.y
                dz = coord_2.z - coord_1.z
                coord_1.x -= dx + random.uniform(-noise, noise)
                coord_1.y -= dy + random.uniform(-noise, noise)
                coord_1.z -= dz + random.uniform(-noise, noise)
                coords.append(coord_1)
                coords, recording = adjust_coords(
                    n_iter=iterations_per_residue,
                    seq_str=seq_str[:len(coords)],
                    input_coords=coords,
                    target_matrix=distance_matrix,
                    temperature=temperature,
                    active_keep_rate=active_keep_rate_spine,
                    max_delta=max_delta,
                    # adjust_last=False,
                )
                full_recording.extend(recording)

        settled_coords, settled_recording = adjust_coords(
            n_iter=iterations_to_settle,
            seq_str=seq_str,
            input_coords=coords,
            target_matrix=distance_matrix,
            temperature=temperature,
            active_keep_rate=active_keep_rate_settle,
            max_delta=max_delta,
            # adjust_last=False,
        )
        full_recording.extend(settled_recording)

        return settled_coords, full_recording


# --------------------------- script entry -------------------------- #

if __name__ == "__main__":
    if TRAIN:
        # delete on demand
        if RESET_DATA:
            _delete_path(DATASET_PATH)
        if RESET_MODEL:
            _delete_path(MODEL_PATH)

        # 1. ===== TRAINING THE MODEL =====
        torch.set_printoptions(precision=4, sci_mode=False)

        model = SpineModelBig()
        model.to(device)
        tr_loader, vl_loader = model.get_dataloaders()

        # ── grab one validation sample ──────────────────────────────────
        sample_x, target = vl_loader.dataset[0]  # (4, L) , (L, L)
        sample_x = sample_x.unsqueeze(0).to(device)  # (1, 4, L)
        target = target.to(device)

        with torch.no_grad():
            out_before = model(sample_x).squeeze(0)  # (L, L)
            mse_before = nn.functional.mse_loss(out_before, target).item()

        # ── train (continues from cached weights if any) ────────────────
        model.fit(tr_loader, vl_loader)

        with torch.no_grad():
            out_after = model(sample_x).squeeze(0)
            mse_after = nn.functional.mse_loss(out_after, target).item()

        # ── nicely formatted report ─────────────────────────────────────
        print(f"\nInput encoding shape : {sample_x.shape[1:]}")  # (4, L)
        print(f"Target distance shape : {target.shape}")  # (L, L)

        print(f"\n>>> TARGET")
        print(target)

        print(f"\n>>> BEFORE training — MSE vs. target = {mse_before:.4f}")
        print(out_before)

        print(f"\n>>> AFTER  training — MSE vs. target = {mse_after:.4f}")
        print(out_after)

        # absolute error matrices (optional, comment out if too verbose)
        # print("\n|before - target| :")
        # print((out_before - target).abs())

        # print("\n|after - target|  :")
        # print((out_after - target).abs())
        # print(model.val_history,model.train_history)
        model.plot_history()
        
    # 2. ===== USING THE MODEL =====
    spine_model = SpineModelBig()
    spine_model.to(device)

    seq_str = "CCCCCCCCCGGGGGGGAAAAAAAAACCCCAAAAGGUUGGUGUUGGUGUGGAGAGAGAGAGUAGAGUAGAG"
    # create a distance matrix for a random sequence
    # distance_matrix = spine_model.construct_distance_matrix("ACGUAAAA")
    spine_coords, recording = spine_model.construct_spine_coords(
        seq_str=seq_str,
    )

    create_video(
        target_coords=spine_coords,
        recording=recording,
        seq_str=seq_str,
        save_path="videos/SpineModelBig.mp4",
        speed=5,
    )

    plot_coords_list(
        seq_str=seq_str, coords=spine_coords
    )

    # plot the coordinates
    # distance_matrix = spine_model.construct_distance_matrix(
    #     seq_str="AAAAAAAAAA"
    # )
    # plot_distance_heatmap(distance_matrix)
    # plot_coords_list([spine_coords], ["Spine Coordinates"])
