import os 
import torch
import torch.nn as nn
from tools import *
from DataTypes import Nucleotide, Vector, Cluster


class Generator(nn.Module):
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