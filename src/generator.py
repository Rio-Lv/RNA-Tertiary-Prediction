import os
import torch
import torch.nn as nn
from tools import *
from DataTypes import Nucleotide, Vector, Cluster
import numpy as np


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
        dx = x[0]
        dy = x[1]
        dz = x[2]
        
        cluster.update(Vector(x=dx, y=dy, z=dz))
        return cluster
    
    def make_clusters(self, n_clusters:int = 10):
        """
        Generate a cluster based on the given sequence.
        """
        # Create a list of nucleotides based on the sequence
        nucleotides = []
        for i in range(n_clusters):
            random_type = np.random.choice(["A", "C", "G", "U", "N"])
            coordinate = Vector(
                x=i * 6.5,
                y=0,
                z=0,
            )
            nucleotides.append(
                Nucleotide(
                    index=i,
                    type=random_type,
                    coordinate=coordinate,
                )
            )
        
        distances = np.zeros((len(nucleotides), len(nucleotides)))

        for i in range(len(nucleotides)):
            for j in range(len(nucleotides)):
                if i != j:
                    dx = nucleotides[i].coordinate.x - nucleotides[j].coordinate.x
                    dy = nucleotides[i].coordinate.y - nucleotides[j].coordinate.y
                    dz = nucleotides[i].coordinate.z - nucleotides[j].coordinate.z
                    distances[i][j] = np.sqrt(dx**2 + dy**2 + dz**2)
       
        # Create clusters based on the distances in groups of 5
        # This is a naive approach, in a real scenario we would use a clustering algorithm
        
        clusters:list[Cluster] = [] # cluster per nucleotide
        cluster_size = 5 # Nearest Neighbors per cluster
        for i in range(len(nucleotides)):
            # Get the indices of the nearest neighbors
            nearest_neighbors = np.argsort(distances[i])[:cluster_size]
            # Create a cluster with the nucleotide and its nearest neighbors
            cluster_nucleotides = [nucleotides[j] for j in nearest_neighbors]
            clusters.append(Cluster(nucleotides=cluster_nucleotides))
            
        n_iter = 10
        for _ in range(n_iter):
            for cluster in clusters:
                cluster = self.forward(cluster)
                # Update the cluster with the new coordinates
        return clusters
        

if __name__ == "__main__":
    # Example usage
    # Assuming you have a Cluster object
    fake_generator = FakeGenerator()
    print(fake_generator)
    clusters = fake_generator.make_clusters(5)
    [print(cluster)  for cluster in clusters]
    
