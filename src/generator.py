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
        clusters.append(Cluster(nucleotides=cluster_nucleotides, real=real))
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
        default_nucleotides = [
            Nucleotide(
                index=i,
                type="A",
                coordinate=Vector(x=0, y=0, z=0),
            )
            for i in range(cluster_size)
        ]
        default_cluster = Cluster(real=False, nucleotides=default_nucleotides)
        input_length = (
            default_cluster.tensor.shape[0] * default_cluster.tensor.shape[1]
        )  # cluster_size * 8
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

    def forward(self, cluster: Cluster):
        """
        Forward pass of the generator.
        """
        # Get the tensor representation of the cluster
        x = cluster.tensor
        # Flatten the tensor to a vector
        x = x.view(-1)
        # Pass through the network
        x = self.stack(x)
        x = x.view(-1, 3)  # Reshape to (cluster_size, 3)
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
        return cluster

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

        clusters = nucleotides_to_clusters(nucleotides, real=False, cluster_size=self.cluster_size)

        n_iter = 1
        for _ in range(n_iter):
            for cluster in clusters:
                cluster = self.forward(cluster)
                # Update the cluster with the new coordinates
        return clusters


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
            clusters += nucleotides_to_clusters(
                nucleotides, real=True, cluster_size=self.cluster_size
            )
        return clusters[:n_clusters]


class Evaluator(nn.Module):
    def __init__(self, cluster_size: int):
        self.cluster_size = cluster_size
        default_nucleotides = [
            Nucleotide(
                index=i,
                type="A",
                coordinate=Vector(x=0, y=0, z=0),
            )
            for i in range(cluster_size)
        ]
        default_cluster = Cluster(real=False, nucleotides=default_nucleotides)
        input_length = (
            default_cluster.tensor.shape[0] * default_cluster.tensor.shape[1]
        )  # cluster_size * 8

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

    def forward(self, cluster: Cluster):
        """
        Forward pass of the evaluator.
        """
        # Get the tensor representation of the cluster
        x = cluster.tensor
        # Flatten the tensor to a vector
        x = x.view(-1)
        # Pass through the network
        x = self.stack(x)
        return x

    def train(self, clusters: list[Cluster]):
        """
        Train the evaluator on the given clusters. use cluster.tensor as input.
        and cluster.real as target.
        cluster tensor is a torch tensor of shape (cluster_size, 8) where each row is
        (dx, dy, dz, a, c, g, u, -, cb).
        """
        
        # Create a tensor from the clusters
        x = torch.stack([cluster.tensor for cluster in clusters])
        y = torch.tensor([1 if cluster.real else 0 for cluster in clusters])
        # Flatten the tensor to a vector
        x = x.view(-1, self.cluster_size * 8)
        # Pass through the network
        x = self.stack(x)
        # Compute the loss
        loss = nn.BCEWithLogitsLoss()(x, y)
        print(f"Loss: {loss.item()}")
        return loss
        

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


if __name__ == "__main__":
    # set file dir as current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    cluster_size = 10

    fake_generator = FakeGenerator(cluster_size=cluster_size)
    real_generator = RealGenerator(cluster_size=cluster_size)
    evaluator = Evaluator(cluster_size=cluster_size)

    clusters = generate_clusters_dataset(
        fake_generator=fake_generator, real_generator=real_generator, n_clusters=500
    )
    
    # evaluator.train(clusters)
    
