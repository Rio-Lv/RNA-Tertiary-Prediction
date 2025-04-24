from Sequence import SequenceDataset
from tools import *
import pandas as pd
sequences_path = "data/train_sequences.csv"
sequences = pd.read_csv(sequences_path)

min_seq_len = 120
# Filter sequences where string length is less than min_seq_len
sequences = sequences[sequences['sequence'].str.len() >= min_seq_len]

seq_id = sequences.iloc[0]['target_id']
seq_str = sequences.iloc[0]['sequence'] # string

seq_ds = SequenceDataset(n_sequences=1, subset_len=200)
seq = seq_ds.get_sequence(seq_id)
plot_coords_mids(seq_str=seq_str,coords=seq.coords, pair_threshold=12)
plot_coords_list(seq_str=seq_str,coords=seq.coords, pair_threshold=10)