import os
import torch
import torch.nn as nn
from tools import *
from DataTypes import Nucleotide, Vector, Cluster


class FakeGenerator(nn.Module):
    def __init__(self):
        default_cluster = Cluster()
        input_length = default_cluster.tensor.shape[0] * default_cluster.tensor.shape[1]
        output_length = 3
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

    def forward(self, cluster: Cluster = Cluster()):
        """
        Forward pass of the generator.
        """
        # Get the tensor representation of the cluster
        x = cluster.tensor
        # Flatten the tensor to a vector
        x = x.view(-1)
        # Pass through the network
        x = self.stack(x)
        print(x)
        dx = x[0]
        dy = x[1]
        dz = x[2]
        
        cluster.update(Vector(dx=dx, dy=dy, dz=dz))
        return cluster
    
    def get_cluster(self, sequence: str = "ACGT"):
        """
        Generate a cluster based on the given sequence.
        """
        # Create a list of nucleotides based on the sequence
        nucleotides = [
            Nucleotide(index=i, type=nt, coordinate=Vector(x=i, y=i, z=i))
            for i, nt in enumerate(sequence)
        ]
        # Create a cluster with the nucleotides
        cluster = Cluster(nucleotides=nucleotides)
        return cluster


if __name__ == "__main__":
    # Example usage
    # Assuming you have a Cluster object
    fake_generator = FakeGenerator()
    print(fake_generator)
