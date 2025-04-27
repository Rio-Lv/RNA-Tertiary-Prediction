""" 
 This model will predict the distance between sequence sections within on molecule.
 lets go with seq len of 10 to 10
 1. Create a dataset of full length sequences
 2. Turn them in 10 + 10 encodings as input and centroid distance as output
 3. Train a model to predict the distance between the two sections
"""
TARGET_LEN = 5
PROCESS_N_SEQS = 100
from Sequence import SequenceDataset, Sequence
from tools import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# --- Helper functions ---
def get_centroid(coords):
    """
    Get the centroid of a list of coordinates
    """
    x = [coord.x for coord in coords]
    y = [coord.y for coord in coords]
    z = [coord.z for coord in coords]
    centroid = (sum(x)/len(x), sum(y)/len(y), sum(z)/len(z))
    return centroid

def get_distance(seq1:Sequence, seq2:Sequence):
    """
    Get the distance between two sequences
    """
    centroid1 = get_centroid(seq1.coords)
    centroid2 = get_centroid(seq2.coords)
    distance = ((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2 + (centroid1[2] - centroid2[2])**2)**0.5
    return distance
    

def generate_data():
    sequences_path = "data/train_sequences.csv"
    sequences = pd.read_csv(sequences_path)

    min_seq_len = TARGET_LEN * 2
    # Filter sequences where string length is less than min_seq_len
    sequences = sequences[sequences['sequence'].str.len() >= min_seq_len]

    seq_id = sequences.iloc[0]['target_id']
    seq_str = sequences.iloc[0]['sequence'] # string

    seq_ds = SequenceDataset()
    pairs = [] # [section1, section 2, distance]
    for i in range(PROCESS_N_SEQS):
        print(f"Processing sequence {i+1}/{PROCESS_N_SEQS}")
        seq_id = sequences.iloc[i]['target_id']
        seq_str = sequences.iloc[i]['sequence'] # string
        seq = seq_ds.get_sequence(seq_id)
        seq_len = len(seq_str)
        
        for j in range(seq_len - TARGET_LEN):
            section_1 = seq.subset(j, j+TARGET_LEN)
            for k in range(j+TARGET_LEN, seq_len - TARGET_LEN):
                section2 = seq.subset(k, k+TARGET_LEN)
                distance = get_distance(section_1, section2)
                pairs.append((section_1.encoding, section2.encoding, distance))
                
    print(f"Generated {len(pairs)} pairs")
    # convert to torch data
    pairs = [(torch.tensor(pair[0]), torch.tensor(pair[1]), torch.tensor(pair[2])) for pair in pairs]
    # convert to DataLoader
    return pairs

if __name__ == "__main__":
    pairs = generate_data()
    print(pairs[0])