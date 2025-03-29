import os
import torch
from torch import Tensor
from tools import *
from strand_generator_crude import crude_simulate as make_fake_nucleotides
from typing import Tuple
import random
# What best format for the Generator and the Discriminator?

# =================== Generate Data ===================
def pdb_id_to_clusters(pdb_id:str, sequences:Sequences,labels:Labels )-> Tuple[list[Nucleotide], list[Nucleotide]]:

    max_nucleotides = 300
    # 1. Use Labels to Generate Real Nucleotides
    strand = grab_strand(pdb_id, labels)
    real_nucleotides = strand_to_nucleotides(strand)[:max_nucleotides]
    real_indices = [nt.index for nt in real_nucleotides]
    
    # 2. Use Sequences to Generate Fake Nucleotides
    sequence = grab_sequence(pdb_id, sequences)
    assert len(sequence.sequence) > 5, "Sequence should be longer than 5"
    fake_nucleotides = sequence_to_nucleotide_line(sequence)[:max_nucleotides]
    fake_nucleotides = make_fake_nucleotides(5, fake_nucleotides, k=2, steps=100)
    fake_nucleotides = [nt for nt in fake_nucleotides if nt.index in real_indices]
    
    assert len(fake_nucleotides) == len(real_nucleotides), "Fake and Real Nucleotides should be the same length"
    
    # 3. Generate Distance Matrix
    distance_matrix_real = calculate_distance_matrix(real_nucleotides)
    distance_matrix_fake = calculate_distance_matrix(fake_nucleotides)
    # 4. Generate Clusters
    
    fake_clusters = []
    for i in range(len(fake_nucleotides)):
        nearest_nucleotides = get_nearest_nucleotides(
            5, fake_nucleotides[i].index, fake_nucleotides, distance_matrix_fake
        )
        tensor_list = []
        for nt in nearest_nucleotides:
            tensor_list.append(nt.get_array())
            
        tensor = Tensor(tensor_list)
        assert tensor.shape == (5, 8), f"Tensor should be of shape (5, 8) not {tensor.shape}"
        fake_clusters.append(tensor)
    real_clusters = []
    for i in range(len(real_nucleotides)):
        nearest_nucleotides = get_nearest_nucleotides(
            5, real_nucleotides[i].index, real_nucleotides, distance_matrix_real
        )
        tensor_list = []
        for nt in nearest_nucleotides:
            tensor_list.append(nt.get_array())
            
        tensor = Tensor(tensor_list)
        assert tensor.shape == (5, 8), f"Tensor should be of shape (5, 8) not {tensor.shape}"
        real_clusters.append(tensor)
    # 5. return clusters
    return fake_clusters, real_clusters
    
def generate_data(n_sequences:int = 5) -> EvaluatorDataset:
    
    sequences_path = "data/train_sequences.csv"
    sequences = Sequences(df=pd.read_csv(sequences_path))
    
    pdb_ids = list(sequences.df["target_id"].unique().tolist())
    random.shuffle(pdb_ids)
 
    labels_path = "data/train_labels.csv"
    labels = Labels(df=pd.read_csv(labels_path))
    
    real_clusters_list = []
    fake_clusters_list = []
    
    curr_index = 1
    for pdb_id in pdb_ids[:n_sequences]:
        print(f"Generating data for {pdb_id} ({curr_index}/{n_sequences}) ")
        if len(pdb_id) > 5:
            real_clusters, fake_clusters = pdb_id_to_clusters(pdb_id, sequences, labels)
            real_clusters_list.extend(real_clusters)
            fake_clusters_list.extend(fake_clusters)
        else:
            print(f"Skipping {pdb_id} as it is too short")
            continue
        curr_index += 1
        
    # 6. Create Dataset
    dataset = EvaluatorDataset(real_clusters_list, fake_clusters_list)
    return dataset
        
    
# ========== Test ==========
if __name__ == "__main__":
    # set dir to file location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # 1. Generate Data
    dataset = generate_data(n_sequences=200)
    dataset.summarise()
    print("Data generated successfully")
    
    # 2. Save Data
    torch.save(dataset, "data/evaluator_dataset.pt")
    # 3. Load Data
    # dataset = torch.load("data/evaluator_dataset.pt")