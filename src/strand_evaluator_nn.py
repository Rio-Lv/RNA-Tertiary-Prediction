import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from tools import *
# import dataset from data/evaluator_dataset.pt

# =================== Model Creation ===================
class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()
        # Flatten the input (5, 7) into a vector of size 35
        self.flatten = nn.Flatten()
        # Define a simple fully connected network
        self.fc = nn.Sequential(
            nn.Linear(35, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Output a single value for binary classification
        )
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # set dir to file location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("============ Strand Evaluator ===========")
    
    # 1. Load the dataset

    dataset:EvaluatorDataset = torch.load("data/evaluator_dataset.pt", weights_only=False)
    print("Data loaded successfully")
    
    # 2. Initialize Model
    model = Evaluator()