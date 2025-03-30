import os
import torch
import torch.nn as nn
from strand_evaluator_nn import EvaluatorModel
import torch.optim as optim
from tools import *

class GeneratorModel(nn.Module):
    def __init__(self):
        super().__init__()
        # We define a network that expects a flattened vector of size 40.
        self.stack = nn.Sequential(
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # Output a delta (dx, dy, dz)
        )
    
    def forward(self, x):
        """
        x: Tensor of shape (5,8) where each row is 
           (dx, dy, dz, a, c, g, u, -).
           
        The generator iteratively computes a delta (dx, dy, dz) from the entire cluster,
        then adds that delta to the first three features of all non-base nucleotides.
        The base nucleotide (first row) remains fixed.
        """
        max_iter = 5
        # Save the base nucleotide (first row) so it remains unchanged
        base = x[0:1, :].clone()  # shape: (1, 8)
        # Start with the initial cluster
        cluster = x.clone()
        for _ in range(max_iter):
            # Flatten the entire cluster into a 40-element vector
            flat = cluster.view(-1)  # shape: (40,)
            delta = self.stack(flat)  # shape: (3,)
            # Update all nucleotides except the base:
            updated_rest = cluster[1:, :].clone()  # shape: (4, 8)
            # Add delta to the first three features (dx, dy, dz) of each nucleotide in updated_rest.
            updated_rest[:, :3] = updated_rest[:, :3] + delta.unsqueeze(0)
            # Reassemble the cluster with the fixed base.
            cluster = torch.cat([base, updated_rest], dim=0)
        return cluster
    
    # ====== Training Criterion =======
    def criterion(self, output, target):
        return 
    
    # ====== Training the Generator =======
    def train():
        return 
    
    
    # ====== Renderable Output =======
    @staticmethod
    def cluster_tensor_to_coordinate(cluster_tensor):
        """
        return Coordinate(x, y, z) of the first nucleotide in the cluster tensor
        cluster_tensor: Tensor of shape (5, 8)
        """
        # Extract the first nucleotide's coordinates
        x = cluster_tensor[0, 0].item()
        y = cluster_tensor[0, 1].item()
        z = cluster_tensor[0, 2].item()
        return Coordinate(x, y, z)

    def sequence_to_pdb(self, sequence_str):
        sequence = Sequence(
            target_id="Made By Generator Model",
            sequence=sequence_str,
        )
        nucleotides = sequence_to_nucleotide_line(sequence)
        clusters = get_nearest_nucleotides_kdtree(nucleotides, 5)
        assert len(clusters) == len(nucleotides), "Clusters should be the same length as nucleotides"
        
        for cluster in clusters:
            coord_move = self.cluster_tensor_to_coordinate(cluster)
            print(coord_move)
      


