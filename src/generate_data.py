import os
from pydantic import BaseModel, field_validator
import torch
from torch import Tensor
from numpy import ndarray
from tools import *
from strand_generator_crude import crude_simulate as make_fake_nucleotides
from torch.utils.data import Dataset
# What best format for the Generator and the Discriminator?

def generate_data()-> Dataset:
    # 1. Use Sequences to Generate Fake Nucleotides
    sequences_path = "data/train_sequences.csv"
    sequences = Sequences(df=pd.read_csv(sequences_path))
    sequence = grab_random_sequence(sequences)

    fake_nucleotides = sequence_to_nucleotide_line(sequence)
    fake_nucleotides = make_fake_nucleotides(5, fake_nucleotides, k=2, steps=100)

    # 2. Use Labels to Generate Real Nucleotides
    labels_path = "data/train_labels.csv"
    labels = Labels(df=pd.read_csv(labels_path))
    pdb_id = sequence.target_id
    strand = grab_strand(pdb_id, labels)
    real_nucleotides = strand_to_nucleotides(strand)
    
    # 3. Generate Distance Matrix
    distance_matrix_fake = calculate_distance_matrix(fake_nucleotides)
    distance_matrix_real = calculate_distance_matrix(real_nucleotides)
    # 4. Generate Clusters
    
    assert len(fake_nucleotides) == len(real_nucleotides), "Fake and Real Nucleotides should be the same length"
    fake_clusters = []
    for i in range(len(fake_nucleotides)):
        nearest_nucleotides = get_nearest_nucleotides(
            5, fake_nucleotides[i].index, fake_nucleotides, distance_matrix_fake
        )
        tensor_list = []
        for nt in nearest_nucleotides:
            tensor_list.append(nt.get_array())
            
        tensor = Tensor(tensor_list)
        assert tensor.shape == (5, 7), f"Tensor should be of shape (5, 7) not {tensor.shape}"
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
        assert tensor.shape == (5, 7), f"Tensor should be of shape (5, 7) not {tensor.shape}"
        real_clusters.append(tensor)
    # 5. Create Dataset
    dataset = EvaluatorDataset(fake_clusters, real_clusters)
    return dataset
    
    
        
    
# ========== Test ==========
if __name__ == "__main__":
    # set dir to file location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # 1. Generate Data
    dataset = generate_data()
    print("Data generated successfully")
    # 2. Check Data
    for i in range(5):
        print("Fake Cluster:", dataset[i][0])
        print("Is Fake:", dataset[i][1])
        print("Real Cluster:", dataset[-i][0])
        print("Is Fake:", dataset[i + 5][1])
    # 3. Check Length
    print("Length of Dataset:", len(dataset))
    # 4. Check Shape
    print("Shape of Fake Cluster:", dataset[0][0].shape)
    print("Shape of Real Cluster:", dataset[-1][0].shape)
    # 5. Check Data Types
    print("Data Type of Fake Cluster:", dataset[0][0].dtype)
    print("Data Type of Real Cluster:", dataset[-1][0].dtype)
    # 6. Save Data
    torch.save(dataset, "data/evaluator_dataset.pt")
    # 7. Load Data
    # dataset = torch.load("data/evaluator_dataset.pt")