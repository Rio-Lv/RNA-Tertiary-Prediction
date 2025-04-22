""" 
This model uses Encodings and Recordings to predict stability.
Where the Earlier in the Recording the Less Stable
"""
import os
import pandas as pd

os.chdir(os.path.dirname(__file__))

LABELS_PATH = "data/train_labels.csv"
SEQUENCE_PATH = "data/train_sequences.csv"

class StabilityDataset:
    def __init__(self):
        self.labels = pd.read_csv(LABELS_PATH)
        self.sequences = pd.read_csv(SEQUENCE_PATH)