import os
import torch
import torch.nn as nn
from tools import *
from DataTypes import Nucleotide, Vector, Cluster
import numpy as np
import pandas as pd
import time


# ---- Helper functions ----
def nucleotides_to_clusters(
    nucleotides: list[Nucleotide], real: bool, cluster_size: int
) -> list[Cluster]:
    """
    Convert a list of nucleotides to clusters.
    """
    distances = np.zeros((len(nucleotides), len(nucleotides)))
    for i in range(len(nucleotides)):
        for j in range(len(nucleotides)):
            if i != j:
                dx = nucleotides[i].coordinate.x - nucleotides[j].coordinate.x
                dy = nucleotides[i].coordinate.y - nucleotides[j].coordinate.y
                dz = nucleotides[i].coordinate.z - nucleotides[j].coordinate.z
                distances[i][j] = np.sqrt(dx**2 + dy**2 + dz**2)

    # Create clusters based on the distances in groups of cluster_size
    # This is a naive approach, in a real scenario we would use a clustering algorithm
    clusters: list[Cluster] = []  # cluster per nucleotide
    for i in range(len(nucleotides)):
        # Get the indices of the nearest neighbors
        nearest_neighbors = np.argsort(distances[i])[:cluster_size]
        # Create a cluster with the nucleotide and its nearest neighbors
        cluster_nucleotides = [nucleotides[j] for j in nearest_neighbors]
        clusters.append(Cluster(nucleotides=cluster_nucleotides, real=real, cluster_size=cluster_size))
    return clusters


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


class FakeGenerator(nn.Module):
    cluster_size: int

    def __init__(self, cluster_size: int):
        self.cluster_size = cluster_size

        input_length = 8 * cluster_size  # dx, dy, dz, a, c, g, u, -, cb

        # cluster_size nucleotides, each with 8 features
        output_length = 3 * cluster_size  # dx, dy, dz

        super().__init__()
        # We define a network that expects a flattened vector of size 40.
        self.stack = nn.Sequential(
            nn.Linear(input_length, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_length),  # Output a delta (dx, dy, dz)
        )

    def make_clusters(self, n_clusters: int):
        """
        Generate a cluster based on the given sequence.
        """
        # Create a list of nucleotides based on the sequence
        nucleotides = []
        for i in range(n_clusters):
            # create random float 0-1
            magnitude = np.random.rand() * 6.5
            # create random rotation
            rx = np.random.rand() * 2 * np.pi
            ry = np.random.rand() * 2 * np.pi
            rz = np.random.rand() * 2 * np.pi
            # create translation from magnitude and rotation
            x = magnitude * np.cos(rx)
            y = magnitude * np.sin(ry)
            z = magnitude * np.sin(rz)

            # make regular float
            x = float(x)
            y = float(y)
            z = float(z)

            random_type = np.random.choice(["A", "C", "G", "U", "N"])
            coordinate = Vector(
                x=x,
                y=y,
                z=z,
            )
            nucleotides.append(
                Nucleotide(
                    index=i,
                    type=random_type,
                    coordinate=coordinate,
                )
            )

        clusters = nucleotides_to_clusters(
            nucleotides, real=False, cluster_size=self.cluster_size
        )

        return clusters

    def forward(self, x):
        """
        Forward pass that supports a batch of flattened cluster tensors.
        x: Tensor of shape (batch_size, 8 * cluster_size)
        Returns: Tensor of shape (batch_size, cluster_size, 3)
        """
        batch_size = x.size(0)
        x = x.clone()
        x = x.view(batch_size, 8 * self.cluster_size)
        x = self.stack(x)  # (batch_size, 3 * cluster_size)
        x = x.view(batch_size, self.cluster_size, 3)
        return x

    def update_cluster(self, cluster: Cluster):
        x = self.forward(cluster.tensor)
        vectors = []
        # Update the coordinates of the nucleotides
        for i in range(len(cluster.nucleotides)):
            nucleotide = cluster.nucleotides[i]
            vector = Vector(
                x=nucleotide.coordinate.x + x[i][0],
                y=nucleotide.coordinate.y + x[i][1],
                z=nucleotide.coordinate.z + x[i][2],
            )
            vectors.append(vector)
        # Update the cluster with the new coordinates
        cluster.update(vectors=vectors)
        cluster.real = False
        return cluster

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
        prediction is closer to 1 (i.e. 'real').
        """
        
        # add noise to the clusters
        for cluster in clusters:
            noise = torch.randn(cluster.tensor.shape) * 3
            cluster.tensor += noise
            
        
        self.train()  # Ensure generator is in train mode
        evaluator.eval()  # Ensure evaluator is in eval mode so its parameters are frozen
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        # Prepare dataset: flatten each cluster tensor (shape: cluster_size*8)
        cluster_tensors = [cluster.tensor.view(-1) for cluster in clusters]
        X = torch.stack(cluster_tensors)  # (N, 8*cluster_size)
        y = torch.ones((X.size(0), 1))  # Target is 1 for all clusters
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            early_stop = False
            for i, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                # Get delta from fake generator: (batch_size, cluster_size, 3)
                delta = self.forward(inputs)
                # Reshape inputs to (batch_size, cluster_size, 8)
                inputs_reshaped = inputs.view(-1, self.cluster_size, 8)
                updated = inputs_reshaped.clone()
                # Add delta to the coordinate columns (first 3 columns)
                updated[:, :, :3] = updated[:, :, :3] + delta
                # Flatten updated tensor back to (batch_size, 8 * cluster_size)
                updated_flat = updated.view(-1, self.cluster_size * 8)

                # Evaluate the updated clusters
                pred = evaluator(updated_flat)
                loss = criterion(pred, targets.float())
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print(
                        f"Fake Generator - Epoch {epoch}, Batch {i}, Loss: {loss.item()}"
                    )
                if loss.item() < loss_cut_off:
                    print(
                        f"Fake Generator - Early stopping at epoch {epoch}, batch {i} with loss {loss.item()}"
                    )
                    early_stop = True
                    break
            if early_stop:
                break

        print("Fake generator training complete.")


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
            while len(sequence_str) > 100 or len(sequence_str) < self.cluster_size:
                pdb_id, sequence_str = self.get_random_sequence()
            # Lets start with smaller clusters
            
            nucleotides = self.pdb_id_to_nucleotides(pdb_id)
            if len(nucleotides) > self.cluster_size:
                clusters += nucleotides_to_clusters(
                    nucleotides, real=True, cluster_size=self.cluster_size
                )
        return clusters[:n_clusters]


class Evaluator(nn.Module):
    def __init__(self, cluster_size: int):
        self.cluster_size = cluster_size
        input_length = 8 * cluster_size  # dx, dy, dz, a, c, g, u, -, cb

        super().__init__()
        # We define a network that expects a flattened vector of size 40.
        self.stack = nn.Sequential(
            nn.Linear(input_length, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Outputs if real or fake
        )

    def forward(self, x):
        """
        Forward pass of the evaluator.
        """
        # Pass through the network
        x = self.stack(x)
        return x

    def train_model(
        self, clusters: list[Cluster], epochs: int, batch_size: int, loss_cut_off: float
    ):
        """
        Train the evaluator on the given clusters.
        """
        self.train()  # Ensure evaluator is in train mode
        # Flatten cluster tensors
        cluster_tensors = [cluster.tensor.view(-1) for cluster in clusters]
        # Check for correct tensor sizes
        for i in range(len(cluster_tensors)):
            if cluster_tensors[i].shape != (self.cluster_size * 8,):
                print(f"Error in cluster {i}: {cluster_tensors[i].shape}")
                break
        targets = [1 if cluster.real else 0 for cluster in clusters]

        # Create dataset and dataloader
        x = torch.stack(cluster_tensors)  # (N, 8*cluster_size)
        y = torch.tensor(targets).view(-1, 1)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Define loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(epochs):
            early_stop = False
            for i, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets.float())
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print(f"Evaluator - Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
                if loss.item() < loss_cut_off:
                    print(
                        f"Evaluator - Early stopping at epoch {epoch}, batch {i} with loss {loss.item()}"
                    )
                    early_stop = True
                    break
            if early_stop:
                break

        print("Evaluator training complete.")
        torch.save(self.state_dict(), "evaluator.pth")


def generate_clusters_dataset(
    fake_generator: FakeGenerator, real_generator: RealGenerator, n_clusters: int
):
    start_time = time.time()

    fake_clusters = fake_generator.make_clusters(n_clusters=n_clusters)
    real_clusters = real_generator.make_clusters(n_clusters=n_clusters)

    [print(cluster) for cluster in real_clusters[:2]]
    [print(cluster) for cluster in fake_clusters[:2]]

    clusters = real_clusters + fake_clusters

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
    # Generate clusters
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

    cluster_size = 10
    n_clusters = 1000
    batch_size = 32
    epochs = 100
    loss_cut_off = 0.001

    fake_generator = FakeGenerator(cluster_size=cluster_size)
    real_generator = RealGenerator(cluster_size=cluster_size)
    evaluator = Evaluator(cluster_size=cluster_size)

    clusters = generate_clusters_dataset(
        fake_generator=fake_generator,
        real_generator=real_generator,
        n_clusters=1000,
    )
    save_clusters_to_csv(clusters=clusters, filename="data/train_clusters.csv")

    # ------ One Round of Training ------
    for i in range(50):
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
