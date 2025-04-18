# SPDX‑License‑Identifier: MIT
"""Train or *continue* training a feed-forward network that maps an encoding
(4 x sequence_size) to a distance matrix (sequence_size x sequence_size).

Key behaviour
-------------
* Always **load** cached model weights _if they exist_, then keep training for
  ``SPINE_TRAIN_EPOCHS`` more iterations.  Each run therefore fine-tunes the
  model a bit further instead of skipping training.
* Two independent toggles control cache deletion:

    RESET_DATA  - delete the cached dataset and regenerate it next run
    RESET_MODEL - delete the cached network weights and start from scratch

Run directly::

    python spine_model.py
"""
from __future__ import annotations

import os, pathlib
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split

# ------------------------- hyper‑parameters ------------------------- #
SPINE_TRAIN_EPOCHS = 4_000
SPINE_MODEL_LR = 0.001
SPINE_TRAIN_BATCH_SIZE = 64**2
SPINE_DATA_TRAIN_FRAC = 0.8
SPINE_WINDOW_SIZE = 6
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
            nn.Linear(4 * sequence_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, sequence_size * sequence_size),
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


# --------------------------- script entry -------------------------- #

if __name__ == "__main__":

    torch.set_printoptions(precision=4, sci_mode=False)

    model = SpineModel()
    train_loader, val_loader = model.get_dataloaders()

    # Always train (continue training if weights were loaded)
    model.fit(train_loader, val_loader)
    model.plot_history()
