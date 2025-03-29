# Assume Input Shape is (5, 8) for each nucleotide
# and the output is a single value (3D coordinates (dx, dy, dz)) 

from tools import *
import os
import torch
import torch.nn as nn

class GeneratorModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten the input (5, 8) into a vector of size 35
        self.flatten = nn.Flatten()
        # Define a simple fully connected network
        self.stack = nn.Sequential(
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # Output a single value for 3D coordinates (dx, dy, dz)
        )
    def forward(self, x):
        """
        x: Tensor of shape (5,8), where each row is (dx, dy, dz, a, c, g, u, -)
        
        The generator outputs a delta (dx, dy, dz) based on the flattened cluster,
        and then updates the dx,dy,dz of all non-base nucleotides (rows 1 to 4)
        iteratively (max_iter times). The base nucleotide (first row) remains fixed.
        """
        max_iter = 5
        # Save the base nucleotide so it remains unchanged
        base = x[0:1, :].clone()  # shape (1,8)
        # Start with the original cluster
        cluster = x.clone()
        for _ in range(max_iter):
            # Compute delta from the current cluster
            flat = self.flatten(cluster)  # shape (40,)
            delta = self.stack(flat)        # shape (3,)
            # Update all nucleotides except the base:
            # Extract the rest (rows 1 to 4)
            updated_rest = cluster[1:, :].clone()  # shape (4,8)
            # Only update the first three features (dx,dy,dz)
            # delta.unsqueeze(0) has shape (1,3) and broadcasts to (4,3)
            updated_rest[:, :3] = updated_rest[:, :3] + delta.unsqueeze(0)
            # Reassemble the cluster with the fixed base
            cluster = torch.cat([base, updated_rest], dim=0)
        return cluster
    
if __name__ == "__main__":
    sample_tensor = torch.randn(5, 8)
    model = GeneratorModel()
    output = model(sample_tensor)
    print("Output:", output)
    print("Output shape:", output.shape)  # Should be (5, 8)
    assert output.shape == (5, 8), f"Output shape should be (5, 8) not {output.shape}"