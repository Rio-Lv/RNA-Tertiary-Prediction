from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from torch.optim import Adam
from Sequence import SequenceDataset

SPINE_TRAIN_EPOCHS = 3500
SPINE_MODEL_LR = 0.01
SPINE_TRAIN_BATCH_SIZE = 64**2
SPINE_DATA_TRAIN_FRAC = 0.7
SPINE_WINDOW_SIZE = 5
SPINE_N_SEQUENCES = 2000

class SpineModel(nn.Module):
    """
    Takes in encoding shape (4, 5) and outputs distance matrix shape (5, 5)
    eg. ----------------------------
    ENCODING:
    tensor([[0., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.]])
    ----------------------------

    DISTANCE MATRIX:
    tensor([[ 0.0000,  6.6771, 11.1824, 14.6738, 16.7382],
            [ 6.6771,  0.0000,  6.8508, 12.7772, 16.7564],
            [11.1824,  6.8508,  0.0000,  6.6834, 11.6652],
            [14.6738, 12.7772,  6.6834,  0.0000,  5.5299],
            [16.7382, 16.7564, 11.6652,  5.5299,  0.0000]])
    """

    def __init__(self, n_sequences: int, sequence_size: int, lr: float ):
        super().__init__()
        self.n_sequences = n_sequences
        self.sequence_size = sequence_size

        self.train_loader, self.test_loader = self.generate_dataloader()
        
        self.model = nn.Sequential(
            nn.Linear(4 * sequence_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.9),
            nn.ReLU(),
            nn.Linear(64, sequence_size * sequence_size),
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.train_loss_history = []
        self.test_loss_history = []
        

    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        :param x: Input tensor of shape (batch_size, 4 * sequence_size)
        :return: Output tensor of shape (batch_size, sequence_size * sequence_size)
        """
        x = x.view(x.size(0), -1)
        x = self.model(x)
        x = x.view(-1, self.sequence_size, self.sequence_size)
        return x
        
        
    def generate_dataloader(self):
        """
        Generate training data for the model.
        1. Take a sequence of length 5
        2. Predict Distance Matrix using Encoding
        """
        seq_dataset = SequenceDataset(
            n_sequences=self.n_sequences, sequence_size=self.sequence_size
        )
        
        # if file does not exis

        sequences = seq_dataset.source_sequences
        [print(seq) for seq in sequences]

        input_list = []
        target_list = []
        for seq in sequences:
            input_list.append(seq.encoding)  # shape (4, 5)
            target_list.append(seq.distance_matrix)  # shape (5, 5)
            
        # Convert to Dataset
        input_tensor = torch.stack(input_list)
        target_tensor = torch.stack(target_list)
        # build a mask that is True for rows that are 100 % finite
        mask_in  = torch.isfinite(input_tensor.reshape(input_tensor.size(0), -1)).all(1)
        mask_tgt = torch.isfinite(target_tensor.reshape(target_tensor.size(0), -1)).all(1)
        
        keep = mask_in & mask_tgt            # keep only samples that are clean

        input_tensor  = input_tensor[keep]
        target_tensor = target_tensor[keep]
        assert not torch.isnan(input_tensor).any(),  "NaN in inputs"
        assert not torch.isinf(input_tensor).any(),  "Inf in inputs"
        assert not torch.isnan(target_tensor).any(), "NaN in targets"
        assert not torch.isinf(target_tensor).any(), "Inf in targets"
        dataset = TensorDataset(input_tensor, target_tensor)

        n_total      = len(dataset)
        n_train      = int(SPINE_DATA_TRAIN_FRAC * n_total)
        n_test       = n_total - n_train

        # random_split keeps the two subsets "views" over the *same* underlying data
        g = torch.Generator().manual_seed(42)
        train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=g)

        train_loader = DataLoader(train_ds,
                                batch_size=SPINE_TRAIN_BATCH_SIZE,
                                shuffle=True)   # shuffle ONLY the train set
        test_loader  = DataLoader(test_ds,
                                batch_size=SPINE_TRAIN_BATCH_SIZE,
                                shuffle=False)  # deterministic order

        return train_loader, test_loader
    
    def train_model(self):
        for epoch in range(SPINE_TRAIN_EPOCHS):

            # ----- TRAIN --------------------------------------------------
            self.train()
            running, n = 0.0, 0
            for x, y in self.train_loader:
                self.optimizer.zero_grad()
                out   = self(x)
                loss  = self.loss_fn(out, y)
                loss.backward()
                self.optimizer.step()

                running += loss.item() * x.size(0)
                n       += x.size(0)

            train_mse = running / n
            self.train_loss_history.append(train_mse)

            # ----- VALIDATE ----------------------------------------------
            self.eval()
            with torch.no_grad():
                running, n = 0.0, 0
                for x, y in self.test_loader:
                    out   = self(x)
                    loss  = self.loss_fn(out, y)
                    running += loss.item() * x.size(0)
                    n       += x.size(0)

            test_mse = running / n
            self.test_loss_history.append(test_mse)

            # ----- logging -----------------------------------------------
            if epoch % 10 == 0:
                print(f"epoch {epoch:>5}/{SPINE_TRAIN_EPOCHS}"
                    f"  train-MSE={train_mse:.4f}"
                    f"  val-MSE={test_mse:.4f}")
 
    def plot_loss(self):
        plt.figure()
        plt.plot(self.train_loss_history, label="train")
        plt.plot(self.test_loss_history,   label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training vs. validation loss")
        plt.ylim(0, 5)                         # keep if you find it helpful
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

# ================ Test 3 Next Residue Model ==============

    spine_model = SpineModel(n_sequences=SPINE_N_SEQUENCES, sequence_size=SPINE_WINDOW_SIZE, lr=SPINE_MODEL_LR)
    # Generate training data
    spine_train_loader = spine_model.train_loader
    spine_test_loader  = spine_model.test_loader
    
    torch.set_printoptions(precision=4, sci_mode=False)

    # ------------------------------------------------------------------
    test_inputs, test_targets = next(iter(spine_test_loader))

    # ------------------------------------------------------------------
    with torch.no_grad():
        output0 = spine_model(test_inputs)
    # ------------------------------------------------------------------
    spine_model.train_model()
    # ------------------------------------------------------------------
    print(">>> BEFORE training")
    print("MSE =", torch.nn.functional.mse_loss(output0, test_targets).item())
    print(output0[0])
    
    print("\n>>> AFTER training")
    spine_model.eval()              # turn off dropout / batch‑norm if you add them later
    with torch.no_grad():           # no need to track gradients during evaluation
        output1 = spine_model(test_inputs)
    print("MSE =", torch.nn.functional.mse_loss(output1, test_targets).item())
    print(output1[0])

    print("\nTarget")
    print(test_targets)
    spine_model.plot_loss()