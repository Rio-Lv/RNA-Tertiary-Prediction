from tools import *

# import tensorflow as tf
from typing import Literal
from pydantic import BaseModel
import random


class Nucleotide(BaseModel):
    type: Literal["A", "C", "G", "U"]


class Coordinate(BaseModel):
    x: float
    y: float
    z: float


def generate_rand_coord(range=100) -> Coordinate:
    x = random.uniform(0, range)
    y = random.uniform(0, range)
    z = random.uniform(0, range)
    return Coordinate(x=x, y=y, z=z)


def generate_random_strand(sequence: Sequence, id="generate_strand") -> Strand:
    nucleotides: list[Nucleotide] = [Nucleotide(type=nt) for nt in sequence.sequence]
    coordinates: list[Coordinate] = [generate_rand_coord() for _ in nucleotides]
    return Strand(
        ID=[id + f"_{i}" for i in range(1, len(nucleotides) + 1)],
        resname=[nt.type for nt in nucleotides],
        resid=[i for i in range(1, len(nucleotides) + 1)],
        x_1=[coord.x for coord in coordinates],
        y_1=[coord.y for coord in coordinates],
        z_1=[coord.z for coord in coordinates],
    )

def generate_line_strand(sequence: Sequence, id="generate_strand") -> Strand:
    nucleotides: list[Nucleotide] = [Nucleotide(type=nt) for nt in sequence.sequence]
    coordinates: list[Coordinate] = [Coordinate(x=0, y=0, z=i*6.5) for i in range(1, len(nucleotides) + 1)]
    return Strand(
        ID=[id + f"_{i}" for i in range(1, len(nucleotides) + 1)],
        resname=[nt.type for nt in nucleotides],
        resid=[i for i in range(1, len(nucleotides) + 1)],
        x_1=[coord.x for coord in coordinates],
        y_1=[coord.y for coord in coordinates],
        z_1=[coord.z for coord in coordinates],
    )

if __name__ == "__main__":
    print("Generating Strand")
    # 1. Select a pbd_id
    pdb_id = "2LKR_A"
    # 2. Load Sequence data
    sequences = Sequences(df=pd.read_csv("data/train_sequences.csv"))
    sequence = grab_sequence(pdb_id, sequences)
    # 3. Generate Fake Strand
    strand = generate_line_strand(sequence)
    # 4. Save to Examine PDB
    pdb = strand_to_pdb(strand)
    save_pdb(pdb, f"data/pdbs/strand_generator_test_{pdb_id}.pdb")