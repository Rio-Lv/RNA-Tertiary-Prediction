from tools import *

# import tensorflow as tf
from typing import Literal, Tuple
from pydantic import BaseModel
import random



class Coordinate(BaseModel):
    x: float
    y: float
    z: float


class Nucleotide(BaseModel):
    type: Literal["A", "C", "G", "U"]
    coordinate: Coordinate = Coordinate(x=0, y=0, z=0)


def generate_rand_coord(range=100) -> Coordinate:
    x = random.uniform(0, range)
    y = random.uniform(0, range)
    z = random.uniform(0, range)
    return Coordinate(x=x, y=y, z=z)


def generate_line(sequence: Sequence, id="generate_strand") -> Tuple[Strand, list[Nucleotide]]:
    nucleotides: list[Nucleotide] = [
        Nucleotide(type=nt, coordinate=Coordinate(x=0, y=0, z=i * 6.5))
        for i, nt in enumerate(sequence.sequence)
    ]
    strand = Strand(
        ID=[id + f"_{i}" for i in range(1, len(nucleotides) + 1)],
        resname=[nt.type for nt in nucleotides],
        resid=[i for i in range(1, len(nucleotides) + 1)],
        x_1=[nt.coordinate.x for nt in nucleotides],
        y_1=[nt.coordinate.y for nt in nucleotides],
        z_1=[nt.coordinate.z for nt in nucleotides],
    )
    return strand, nucleotides

if __name__ == "__main__":
    print("Generating Strand")
    # 1. Select a pbd_id
    pdb_id = "2LKR_A"
    # 2. Load Sequence data
    sequences = Sequences(df=pd.read_csv("data/train_sequences.csv"))
    sequence = grab_sequence(pdb_id, sequences)
    # 3. Generate Fake Strand
    strand, nucleotides = generate_line(sequence)
    # 4. Save to Examine PDB
    pdb = strand_to_pdb(strand)
    print(pdb)
    path = f"data/pdbs/strand_generator_test_{pdb_id}.pdb"
    save_pdb(pdb, path)
    print("Saved PDB to: ", path)
