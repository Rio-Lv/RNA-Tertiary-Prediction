import os
import torch
import torch.nn as nn
from tools import *
from DataTypes import Nucleotide, Vector, Cluster
import numpy as np
import pandas as pd
import time
import random
import math
from torch import optim

# ---- Use Torch for GPU acceleration ----
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")


# ---- Helper functions ----
import numpy as np


def nucleotides_to_clusters(
    nucleotides: list[Nucleotide], real: bool, cluster_size: int
) -> list[Cluster]:
    """
    Convert a list of nucleotides to clusters using optimized vectorized operations.
    """
    # Extract coordinates to a NumPy array (shape: [n_nucleotides, 3])
    coords = np.array(
        [[n.coordinate.x, n.coordinate.y, n.coordinate.z] for n in nucleotides]
    )

    # Compute pairwise squared distances (avoids sqrt for efficiency)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    squared_dists = np.square(diff).sum(axis=-1)

    # Find nearest neighbors using argpartition (O(n) per row instead of O(n log n))
    # Get indices of cluster_size closest neighbors (including self)
    nearest_indices = np.argpartition(squared_dists, cluster_size - 1, axis=1)[
        :, :cluster_size
    ]

    # Create row indices for advanced indexing
    rows = np.arange(squared_dists.shape[0])[:, np.newaxis]

    # Sort just the nearest indices by distance
    sorted_within = np.argsort(squared_dists[rows, nearest_indices], axis=1)
    nearest_indices = nearest_indices[rows, sorted_within]

    # Convert indices to clusters
    return [
        Cluster(
            nucleotides=[nucleotides[j] for j in row_indices],
            real=real,
            cluster_size=cluster_size,
        )
        for row_indices in nearest_indices
    ]


def save_clusters_to_csv(clusters: list[Cluster], filename: str):
    """
    Save clusters to a CSV file. Add Cluster ID to the first column.
    """

    # Helper function to convert tensors to floats
    def to_float(x):
        return x.item() if hasattr(x, "item") else x

    data = []
    for i, cluster in enumerate(clusters):
        array = cluster.get_array()
        for j in range(len(array)):
            # Ensure that if any element is a Tensor, we convert it to float
            row_values = [to_float(val) for val in array[j]]
            is_real = 1 if cluster.real else 0
            row = [i] + row_values + [is_real]
            data.append(row)

    df = pd.DataFrame(
        data, columns=["Cluster ID", "dx", "dy", "dz", "A", "C", "G", "U", "CB", "real"]
    )
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows to {filename}")

class RealGenerator:
    labels_path = "data/train_labels.csv"
    sequences_path = "data/train_sequences.csv"
    labels: pd.DataFrame
    sequences: pd.DataFrame
    cluster_size: int

    def __init__(self, cluster_size: int):
        self.labels = pd.read_csv(self.labels_path).dropna()
        self.sequences = pd.read_csv(self.sequences_path)
        self.n_sequences = len(self.sequences)
        self.cluster_size = cluster_size

    def get_random_sequence(self):
        n_sequences = len(self.sequences)
        random_sequence = np.random.randint(0, n_sequences)
        sequence = self.sequences.iloc[random_sequence]
        sequence = sequence.values
        pdb_id = sequence[0]
        sequence_str = sequence[1]
        return pdb_id, sequence_str

    def pdb_id_to_nucleotides(self, pdb_id: str):
        """
        Convert a PDB ID to a list of nucleotides.
        """
        # grab labels where ID contains pdb_id
        labels = self.labels[self.labels["ID"].str.contains(pdb_id)]

        nucleotides = []
        # iterate over the labels and create nucleotides
        for index, row in labels.iterrows():
            nucleotide = Nucleotide(
                index=row["resid"],
                type=row["resname"],
                coordinate=Vector(
                    x=row["x_1"],
                    y=row["y_1"],
                    z=row["z_1"],
                ),
            )
            nucleotides.append(nucleotide)
        return nucleotides

    def make_clusters(self, n_clusters: int):
        clusters = []
        while len(clusters) < n_clusters:
            pdb_id, sequence_str = self.get_random_sequence()
            while len(sequence_str) > 300 or len(sequence_str) < self.cluster_size:
                pdb_id, sequence_str = self.get_random_sequence()
            # Lets start with smaller clusters

            nucleotides = self.pdb_id_to_nucleotides(pdb_id)
            if len(nucleotides) > self.cluster_size:
                clusters += nucleotides_to_clusters(
                    nucleotides, real=True, cluster_size=self.cluster_size
                )

        return clusters[:n_clusters]


class FakeGenerator(nn.Module):
    cluster_size: int
    lr: float

    def __init__(self, cluster_size: int, lr: float = 0.001):
        super().__init__()
        self.cluster_size = cluster_size
        self.lr = lr

        # The input will be reshaped to (batch_size, 1, cluster_size, 8)
        # Our goal is to output a delta vector per nucleotide,
        # i.e. an output shape of (batch_size, cluster_size, 3)

        # Define a convolutional block.
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2),
        )

        # Use adaptive pooling to force a fixed output size.
        # For example, regardless of the input spatial dimensions,
        # we output a feature map of size (4,4).
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # which flattens to 32*4*4 = 512.
        self.fc_block = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1), 
            nn.Linear(64, cluster_size * 3),  # One delta (dx,dy,dz) per nucleotide
        )

    def forward(self, x):
        """
        Forward pass of the fake generator.
        x: Tensor of shape (batch_size, 8 * cluster_size)
        Returns: Tensor of shape (batch_size, cluster_size, 3)
        """
        batch_size = x.size(0)
        # Reshape the flattened vector to a 4D tensor: (batch_size, 1, cluster_size, 8)
        x = x.view(batch_size, 1, self.cluster_size, 8)
        x = self.conv_block(x)
        x = self.adaptive_pool(x)  # Now x has shape (batch_size, 32, 4, 4)
        x = x.view(batch_size, -1)  # Flatten to (batch_size, 512)
        x = self.fc_block(x)  # (batch_size, cluster_size * 3)
        x = x.view(
            batch_size, self.cluster_size, 3
        )  # Reshape to (batch_size, cluster_size, 3)
        return x

    def update_cluster(self, cluster: Cluster):
        """
        Inference method: given one Cluster, apply the generator’s delta
        to update its nucleotide coordinates.
        """
        # Ensure the cluster.tensor is flattened as expected.
        delta = self.forward(cluster.tensor.view(1, -1))  # (1, cluster_size, 3)
        delta = delta.view(self.cluster_size, 3)  # (cluster_size, 3)
        vectors = []
        for i in range(self.cluster_size):
            vector = Vector(
                x=delta[i][0].item(),
                y=delta[i][1].item(),
                z=delta[i][2].item(),
            )
            vectors.append(vector)
        cluster.update(vectors)
        return cluster

    def make_clusters(self, n_clusters: int):
        """
        Generate a set of clusters using a simple cumulative translation.
        Each nucleotide’s coordinate is generated based on a random magnitude and rotation.
        Then, clusters are created based on the nearest neighbors and updated using this generator.
        """
        nucleotides = []
        x, y, z = 0, 0, 0
        for i in range(n_clusters):
            magnitude = np.random.rand() * 6.5
            rx = np.random.rand() * 2 * np.pi
            ry = np.random.rand() * 2 * np.pi
            rz = np.random.rand() * 2 * np.pi
            x += float(magnitude * np.cos(rx))
            y += float(magnitude * np.sin(ry))
            z += float(magnitude * np.sin(rz))
            random_type = np.random.choice(["A", "C", "G", "U", "N"])
            coordinate = Vector(x=x, y=y, z=z)
            nucleotides.append(
                Nucleotide(index=i, type=random_type, coordinate=coordinate)
            )

        clusters = nucleotides_to_clusters(
            nucleotides, real=False, cluster_size=self.cluster_size
        )
        updated_clusters = []
        for cluster in clusters:
            updated_clusters.append(self.update_cluster(cluster))
        return updated_clusters


    def train_model(
        self,
        clusters: list[Cluster],
        evaluator: nn.Module,
        epochs: int,
        batch_size: int,
        loss_cut_off: float,
    ):
        """
        Train the fake generator so that when it updates a cluster, the evaluator's
        prediction approaches 1 (i.e. the evaluator believes the cluster is real).
        This method uses a differentiable update mechanism on the underlying cluster tensor.
        """
        # Use only clusters labeled as fake.
        clusters = [c for c in clusters if not c.real]
        self.train()  # Generator in train mode.
        evaluator.eval()  # Evaluator in eval (frozen) mode.
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        # Prepare dataset: flatten each cluster tensor (shape: cluster_size*8)
        cluster_tensors = [cluster.tensor.view(-1) for cluster in clusters]
        X = torch.stack(cluster_tensors)  # Shape: (N, 8 * cluster_size)
        y = torch.ones((X.size(0), 1))  # Target is 1 for each cluster.
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for i, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                # Get delta from the generator: (batch_size, cluster_size, 3)
                delta = self.forward(inputs)
                # Reshape inputs to (batch_size, cluster_size, 8)
                inputs_reshaped = inputs.view(-1, self.cluster_size, 8)
                updated = inputs_reshaped.clone()
                # Update only the coordinate columns (first 3 columns) with the computed delta.
                updated[:, :, :3] = updated[:, :, :3] + delta
                # Flatten updated tensor back to (batch_size, 8 * cluster_size)
                updated_flat = updated.view(-1, self.cluster_size * 8)
                # Evaluate the updated clusters using the evaluator.
                pred = evaluator(updated_flat)
                loss = criterion(pred, targets.float())
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
            avg_loss = epoch_loss / batch_count
            print(f"Fake Generator - Epoch {epoch} Average Loss: {avg_loss:.4f}")
            if avg_loss < loss_cut_off:
                print(f"Fake Generator - Early stopping after epoch {epoch} with average loss {avg_loss:.4f}")
                break


        print("Fake generator training complete.")

    def save(self, filename: str):
        """
        Save the generator model to a file.
        """
        torch.save(self.state_dict(), filename)
        print(f"Saved generator model to {filename}")




class Evaluator(nn.Module):
    cluster_size: int
    lr: float

    def __init__(self, cluster_size: int, lr: float = 0.001):
        """
        The evaluator is now defined as a convolutional network.
        It accepts an input of shape (batch_size, cluster_size * 8) and
        reshapes it to (batch_size, 1, cluster_size, 8). The conv layers
        extract spatial features and an adaptive pooling layer ensures a fixed output size.
        """
        super().__init__()
        self.cluster_size = cluster_size
        self.lr = lr
        # Convolutional block: treat cluster data as a 2D image with 1 channel.
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2),
        )
        # Use adaptive pooling to force a fixed spatial size (e.g. 4x4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        # Fully-connected block: flattened feature vector size will be 16 * 4 * 4 = 512.
        self.fc_block = nn.Sequential(
            nn.Linear(256, 64), 
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.3), 
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Forward pass of the evaluator.
        Expects x to be of shape (batch_size, cluster_size*8).
        Reshape x to (batch_size, 1, cluster_size, 8) before applying conv layers.
        """
        batch_size = x.size(0)
        # Reshape to (batch_size, 1, cluster_size, 8)
        x = x.view(batch_size, 1, self.cluster_size, 8)
        x = self.conv_block(x)
        x = self.adaptive_pool(x)  # Now x has shape (batch_size, 32, 4, 4)
        x = x.view(batch_size, -1)  # Flatten to (batch_size, 512)
        x = self.fc_block(x)
        return x

    def train_model(
        self, clusters: list[Cluster], epochs: int, batch_size: int, loss_cut_off: float
    ):
        """
        Train the evaluator on the given clusters.
        Each cluster provides a tensor (of shape (cluster_size,8)) that is flattened to (cluster_size*8,)
        and paired with a target (1 if real else 0).
        Early stopping is applied when a batch loss is below loss_cut_off.
        """
        self.train()  # Set evaluator to train mode

        # Flatten cluster tensors and check sizes.
        cluster_tensors = [cluster.tensor.view(-1) for cluster in clusters]
        for i in range(len(cluster_tensors)):
            if cluster_tensors[i].shape != (self.cluster_size * 8,):
                print(f"Error in cluster {i}: {cluster_tensors[i].shape}")
                break

        # Use targets (you might use smoothed labels if desired)
        targets = [1 if cluster.real else 0 for cluster in clusters]

        # Build dataset and dataloader
        x = torch.stack(cluster_tensors)  # (N, cluster_size*8)
        y = torch.tensor(targets).view(-1, 1)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for i, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets.float())
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            print(f"Evaluator - Epoch {epoch} average loss: {avg_loss}")
            if avg_loss < loss_cut_off:
                print(f"Evaluator - Early stopping after epoch {epoch} with average loss: {avg_loss}")
                break

    def save(self, filename: str):
        """
        Save the evaluator model.
        """
        torch.save(self.state_dict(), filename)
        print(f"Saved evaluator model to {filename}")

    def eval_cluster(self, cluster: Cluster):
        """
        Evaluate a single cluster.
        """
        # Flatten the cluster tensor and add batch dimension
        x = cluster.tensor.view(1, -1)
            # Get the raw logits from the evaluator
        logits = self(x)
        # Apply sigmoid to get a probability between 0 and 1
        prob = torch.sigmoid(logits)
        # Return the probability (as a float)
        return prob.item()


def generate_clusters_dataset(
    fake_generator: FakeGenerator, real_generator: RealGenerator, n_clusters: int
):
    start_time = time.time()

    fake_clusters = fake_generator.make_clusters(n_clusters=n_clusters)
    real_clusters = real_generator.make_clusters(n_clusters=n_clusters)

    [print(cluster) for cluster in real_clusters[:2]]
    [print(cluster) for cluster in fake_clusters[:2]]

    clusters = real_clusters + fake_clusters
    random.shuffle(clusters)

    print("--- %s seconds ---" % (time.time() - start_time))

    return clusters


def train_round(
    fake_generator: FakeGenerator,
    real_generator: RealGenerator,
    evaluator: Evaluator,
    n_clusters: int,
    epochs: int,
    batch_size: int,
    loss_cut_off: float,
):
    # Generate clusters Initially
    clusters = generate_clusters_dataset(
        fake_generator=fake_generator,
        real_generator=real_generator,
        n_clusters=n_clusters,
    )
    # Train evaluator
    evaluator.train_model(
        clusters, epochs=epochs, batch_size=batch_size, loss_cut_off=loss_cut_off
    )
    # Train fake generator
    fake_generator.train_model(
        clusters=clusters,
        evaluator=evaluator,
        epochs=epochs,
        batch_size=batch_size,
        loss_cut_off=loss_cut_off,
    )


if __name__ == "__main__":
    # set file dir as current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Check Point for Hyperparameters
    cluster_size = 4
    batch_size = 256
    n_clusters = 256*10
    epochs = 200
    n_rounds = 20
    loss_cut_off = 0.01
    lr = 0.005  # Can be changed for refinement?

    fake_generator = FakeGenerator(cluster_size=cluster_size, lr=lr)
    real_generator = RealGenerator(cluster_size=cluster_size)
    evaluator = Evaluator(cluster_size=cluster_size, lr=lr)

    # # --- Load Models to continue training ---
    # if os.path.exists("models/fake_generator.pt"):
    #     print("Loading fake generator model...")
    #     fake_generator.load_state_dict(torch.load("models/fake_generator.pt"))
    # if os.path.exists("models/evaluator.pt"):
    #     print("Loading evaluator model...")
    #     evaluator.load_state_dict(torch.load("models/evaluator.pt"))

    # ------ One Round of Training ------
    for i in range(n_rounds):
        print(f" --- Round {i} --- ")

        train_round(
            fake_generator=fake_generator,
            real_generator=real_generator,
            evaluator=evaluator,
            n_clusters=n_clusters,
            epochs=epochs,
            batch_size=batch_size,
            loss_cut_off=loss_cut_off,
        )
   
        # Save models
        fake_generator.save(f"models/fake_generator.pt")
        evaluator.save(f"models/evaluator.pt")

    fake_generator.save(f"models/fake_generator.pt")
    evaluator.save(f"models/evaluator.pt")

    # TODO: Remove 100 Cap on taking in real sequences
