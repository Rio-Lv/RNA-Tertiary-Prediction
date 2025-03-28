from pydantic import BaseModel, field_validator
from torch import Tensor  
from numpy import ndarray
from tools import *
from strand_generator_crude import crude_simulate as make_fake_nucleotides
# What best format for the Generator and the Discriminator?

class Cluster(BaseModel):
    tensor: Tensor
    @field_validator("tensor", mode="before")
    def check_shape(cls, v: Tensor) -> Tensor:
        """
        Tensor shape should be (5, 7)
        each row is a nucleotide resprenting [x,y,z,A,U,C,G]
        for self and 4 nearest neighbors
        """
        assert v.shape == (5, 7), f"Tensor shape should be (5, 7), got {v.shape}"
        return v
    class Config:
        arbitrary_types_allowed = True

    
class EvalInput(BaseModel):
    real: Cluster
    fake: Cluster
    
def get_cluster(index:int, distance_matrix:ndarray, nucleotides:list[Nucleotide]) -> Cluster:

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
    

    
# ========== Test ==========
if __name__ == "__main__":
    # Test Cluster
    cluster = Cluster(
        tensor=Tensor(
            [
                [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                [4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0],
                [7.0, 8.0, 9.0, 0.0, 0.0, 1.0, 0.0],
                [10.0, 11.0, 12.0, 1.0, 1.0, 1.0, 1.0],
                [13.0, 14.0, 15.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
    )
    print(cluster)