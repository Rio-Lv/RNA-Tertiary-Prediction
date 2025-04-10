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
    
# ---- HYPERPARAMS ----
DROPOUT = 0.1
CLUSTER_SIZE = 7
BATCH_SIZE = 256
N_CLUSTERS = 128 # will be like x8 for different cluster generators
EPOCHS = 10
N_ROUNDS = 200
LOSS_CUT_OFF = 0.01
LR = 0.001  # Can be changed for refinement?
N_ITER = 10 # number of iterations to apply delta update
LOAD_PRETRAINED = False
NOISE_L = 24
NOISE_M = 8
NOISE_S = 1

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

# ---- Create random nucleotides, random relative to last points ----

def create_random_nucleotides(n_clusters, sequence:str = None):
    """
    Create a random set of nucleotides.
    """
    if sequence:
        assert len(sequence) == n_clusters, "Sequence length must match number of clusters."
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
        res_type = None
        if sequence:
            res_type = sequence[i]
        else:
            random_type = np.random.choice(["A", "C", "G", "U", "N"])
            res_type = random_type
        coordinate = Vector(x=x, y=y, z=z)
        nucleotides.append(Nucleotide(index=i, type=res_type, coordinate=coordinate))
    return nucleotides


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

    def make_clusters(self, n_clusters: int, noise: float = None)-> list[Cluster]:
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
        if noise:
            for cluster in clusters:
                vectors = []
                for i in range(self.cluster_size):
                    dx = np.random.uniform(-noise, noise)
                    dy = np.random.uniform(-noise, noise)
                    dz = np.random.uniform(-noise, noise)
                    vectors.append(Vector(x=dx, y=dy, z=dz))
                cluster.update(vectors)
                cluster.real = False
        return clusters[:n_clusters]


class Adjuster(nn.Module):
    cluster_size: int
    lr: float
    n_iter: int

    def __init__(self, cluster_size: int,  n_iter: int=5, lr: float = 0.001):
        super().__init__()
        self.cluster_size = cluster_size
        self.lr = lr
        self.n_iter = n_iter


# Convolutional block with reduced channels, 2x2 kernels, and added padding.
        self.conv_block = nn.Sequential(
            # Padding=1 prevents the spatial dimensions from collapsing too quickly.
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(2, 2), padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(2, 2), padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Adaptive pooling to force a fixed output size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Expected output: (batch, 8, 4, 4)

        # Fully connected block with reduced hidden dimensions.
        self.fc_block = nn.Sequential(
            nn.Linear(8 * 4 * 4, 64),  # 8*4*4=128 features.
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Linear(32, cluster_size * 3),  # Final mapping.
        )


    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 8 * cluster_size)
        Returns: Tensor of shape (batch_size, cluster_size, 3) Delta vectors.
        """
        batch_size = x.size(0)
        # Reshape the flattened vector to a 4D tensor: (batch_size, 1, cluster_size, 8)
        x = x.view(batch_size, 1, self.cluster_size, 8)
        x = self.conv_block(x)
        x = self.adaptive_pool(x)  # Fixed shape: (batch_size, 8, 4, 4)
        x = x.view(batch_size, -1)  # Flatten to (batch_size, 128)
        x = self.fc_block(x)  # Output shape: (batch_size, cluster_size * 3)
        x = x.view(batch_size, self.cluster_size, 3)  # Reshape to (batch_size, cluster_size, 3)
        return x

    def update_cluster(self, cluster: Cluster)-> Cluster:
        """
        Inference method: given one Cluster, apply the generator’s delta
        to update its nucleotide coordinates.
        apply delta over a few steps eg. 4
        """
        n_iter = self.n_iter
        # Start with the initial inputs.
        input_tensor = cluster.tensor.view(1,-1)  # shape: ( 1, cluster_size * 8)
        updated_inputs = input_tensor.clone()  # shape: ( 1, cluster_size * 8)
        # Apply the update repeatedly.
        for _ in range(n_iter):
            # Compute the delta from the generator
            # delta shape: (1, cluster_size, 3)
            delta = self.forward(updated_inputs)

            # Reshape updated_inputs to (1, cluster_size, 8)
            inputs_reshaped = updated_inputs.view(-1, self.cluster_size, 8)

            # Create an updated version (copy) of the reshaped tensor
            updated = inputs_reshaped.clone()

            # Add the computed delta to the coordinate columns (first 3 columns)
            updated[:, :, :3] = updated[:, :, :3] + delta

            # Flatten back to (1, cluster_size * 8) for the next iteration
            updated_inputs = updated.view(-1, self.cluster_size * 8)
        # Reshape the updated inputs back to the cluster tensor shape
        updated_tensor = updated_inputs.view(-1, 8)
        # Grab deltas as Vectors
        deltas = []
        for i in range(self.cluster_size):
            dx = updated_tensor[i][0]
            dy = updated_tensor[i][1]
            dz = updated_tensor[i][2]
            deltas.append(Vector(x=dx, y=dy, z=dz))
        # Update the cluster with the new coordinates
        cluster.update(deltas)
        
        return cluster

    def make_clusters(
        self, n_clusters: int, denoise: bool = False, noise: float = None
    )-> list[Cluster]:
        """
        Generate a set of clusters using a simple cumulative translation.
        Each nucleotides coordinate is generated based on a random magnitude and rotation.
        Then, clusters are created based on the nearest neighbors and updated using this generator.

        or

        Get Real Clusters, add noise then update them to remove the noise.
        """
        if not denoise:
            nucleotides = create_random_nucleotides(n_clusters)

            clusters = nucleotides_to_clusters(
                nucleotides, real=False, cluster_size=self.cluster_size
            )
            updated_clusters = []
            for cluster in clusters:
                updated_clusters.append(self.update_cluster(cluster))
            return updated_clusters
        else:
            # Generate real clusters
            real_generator = RealGenerator(cluster_size=self.cluster_size)
            clusters = real_generator.make_clusters(n_clusters=n_clusters, noise=noise)
            # Update them to remove the noise
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
        self.train()  # Generator in train mode.
        # Use only clusters labeled as fake.
        clusters = [c for c in clusters if not c.real]
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

        n_iter = self.n_iter  # number of iterations to apply delta update

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for i, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()

                # Start with the initial inputs.
                updated_inputs = inputs.clone()  # shape: (batch_size, cluster_size * 8)

                # Apply the update repeatedly.
                for _ in range(n_iter):
                    # Compute the delta from the generator
                    # delta shape: (batch_size, cluster_size, 3)
                    delta = self.forward(updated_inputs)

                    # Reshape updated_inputs to (batch_size, cluster_size, 8)
                    inputs_reshaped = updated_inputs.view(-1, self.cluster_size, 8)

                    # Create an updated version (copy) of the reshaped tensor
                    updated = inputs_reshaped.clone()

                    # Add the computed delta to the coordinate columns (first 3 columns)
                    updated[:, :, :3] = updated[:, :, :3] + delta

                    # Flatten back to (batch_size, cluster_size * 8) for the next iteration
                    updated_inputs = updated.view(-1, self.cluster_size * 8)

                # Evaluate the final updated inputs after all iterations
                pred = evaluator(updated_inputs)
                loss = criterion(pred, targets.float())
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            print(f"Adjuster - Epoch {epoch} - Average Loss: {avg_loss:.4f}")
            if avg_loss < loss_cut_off:
                print(
                    f"Adjuster - Early stopping after epoch {epoch} with average loss {avg_loss:.4f}"
                )
                break

        print("Fake generator training complete.")

    def save(self, filename: str):
        """
        Save the generator model to a file.
        """
        torch.save(self.state_dict(), filename)
        print(f"Saved generator model to {filename}")
        
    def load(self, filename: str):
        """
        Load the generator model from a file.
        """
        if os.path.exists(filename):
            self.load_state_dict(torch.load(filename))
            print(f"Loaded generator model from {filename}")
        else:
            print(f"File {filename} does not exist. Cannot load model.")
        return self


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
            nn.Dropout(DROPOUT),
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
            print(
                f"Evaluator - Epoch {epoch} - average loss: {avg_loss}"
            )
            if avg_loss < loss_cut_off:
                print(
                    f"Evaluator - Early stopping after epoch {epoch} with average loss: {avg_loss}"
                )
                break
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

    def save(self, filename: str):
        """
        Save the evaluator model.
        """
        torch.save(self.state_dict(), filename)
        print(f"Saved evaluator model to {filename}")
        
    def load(self, filename: str):
        """
        Load the evaluator model from a file.
        """
        if os.path.exists(filename):
            self.load_state_dict(torch.load(filename))
            print(f"Loaded evaluator model from {filename}")
        else:
            print(f"File {filename} does not exist. Cannot load model.")
        return self




def generate_clusters_dataset(
    adjuster: Adjuster, real_generator: RealGenerator, n_clusters: int
):
    start_time = time.time()

    # -------- Form Of Real Clusters --------
    real_clusters = real_generator.make_clusters(n_clusters=n_clusters * 6)

    real_clusters_noisy_small = real_generator.make_clusters(
        n_clusters=n_clusters, noise=1
    )
    real_clusters_noisy_medium = real_generator.make_clusters(
        n_clusters=n_clusters, noise=4
    )
    real_clusters_noisy_big = real_generator.make_clusters(
        n_clusters=n_clusters, noise=8
    )
    # -------- Forms Of Fake Clusters --------
    
    fake_clusters = adjuster.make_clusters(n_clusters=n_clusters)

    fake_clusters_denoise_small = adjuster.make_clusters(
        n_clusters=n_clusters, denoise=True, noise=NOISE_S
    )
    fake_clusters_denoise_medium = adjuster.make_clusters(
        n_clusters=n_clusters, denoise=True, noise=NOISE_M
    )
    fake_clusters_denoise_large = adjuster.make_clusters(
        n_clusters=n_clusters, denoise=True, noise=NOISE_L
    )

    # Real as Base
    print("------ Real Clusters ( from database )------")
    [print(cluster) for cluster in real_clusters[:1]]
    print(f"------ Real Clusters ( from database + noise ({NOISE_S}) ) ------")
    [print(cluster) for cluster in real_clusters_noisy_small[:1]]
    print(f"------ Real Clusters ( from database + noise ({NOISE_M}) ) ------")
    [print(cluster) for cluster in real_clusters_noisy_medium[:1]]
    print(f"------ Real Clusters ( from database + noise ({NOISE_L}) ) ------")
    [print(cluster) for cluster in real_clusters_noisy_big[:1]]
    # Fake as Base
    print("------ Fake Clusters ( from random + adjust ) ------")
    [print(cluster) for cluster in fake_clusters[:1]]
    print(f"------ Fake Clusters ( from database + noise ({NOISE_S}) + adjust ) ------")
    [print(cluster) for cluster in fake_clusters_denoise_small[:1]]
    print(f"------ Fake Clusters ( from database + noise ({NOISE_M}) + adjust ) ------")
    [print(cluster) for cluster in fake_clusters_denoise_medium[:1]]
    print(f"------ Fake Clusters ( from database + noise ({NOISE_L}) + adjust ) ------")
    [print(cluster) for cluster in fake_clusters_denoise_large[:1]]

    # -------- Combine Clusters --------
    # primary set
    clusters = real_clusters + fake_clusters
    # add noise to real clusters
    clusters += real_clusters_noisy_big
    clusters += real_clusters_noisy_medium
    clusters += real_clusters_noisy_small
    # add noise to fake clusters
    clusters += fake_clusters_denoise_large
    clusters += fake_clusters_denoise_medium
    clusters += fake_clusters_denoise_small

    random.shuffle(clusters)

    print("--- %s seconds ---" % (time.time() - start_time))

    return clusters


if __name__ == "__main__":
    # set file dir as current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


    adjuster = Adjuster(cluster_size=CLUSTER_SIZE, n_iter=N_ITER, lr=LR)
    real_generator = RealGenerator(cluster_size=CLUSTER_SIZE)
    evaluator = Evaluator(cluster_size=CLUSTER_SIZE, lr=LR)

    # --- Load Models to continue training ---
    if LOAD_PRETRAINED:
        adjuster.load("models/adjuster.pt")
        evaluator.load("models/evaluator.pt")
    
    # ------ Init Dataset ------
    # Generate clusters Initially
    clusters = generate_clusters_dataset(
        adjuster=adjuster,
        real_generator=real_generator,
        n_clusters=N_CLUSTERS,
    )

    # ------ One Round of Training ------
    for i in range(N_ROUNDS):
        print(f" --- Round {i} --- ")

        # Train evaluator
        evaluator.train_model(
            clusters, epochs=EPOCHS, batch_size=BATCH_SIZE, loss_cut_off=LOSS_CUT_OFF
        )
        # Train fake generator
        adjuster.train_model(
            clusters=clusters,
            evaluator=evaluator,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            loss_cut_off=LOSS_CUT_OFF,
        )
        if i % 5 == 0:
            clusters = generate_clusters_dataset(
                adjuster=adjuster,
                real_generator=real_generator,
                n_clusters=N_CLUSTERS,
            )

            # Save models
            adjuster.save(f"models/adjuster.pt")
            evaluator.save(f"models/evaluator.pt")

    adjuster.save(f"models/adjuster.pt")
    evaluator.save(f"models/evaluator.pt")

    # TODO: Remove 100 Cap on taking in real sequences
