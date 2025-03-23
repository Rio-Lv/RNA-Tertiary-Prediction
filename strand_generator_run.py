from tools import *

# import tensorflow as tf
from typing import Literal, Tuple
from pydantic import BaseModel
import random
import numpy as np
from numpy import ndarray
from strand_generator_crude import crude_simulate


def generate_rand_coord(range=100) -> Coordinate:
    x = random.uniform(0, range)
    y = random.uniform(0, range)
    z = random.uniform(0, range)
    return Coordinate(x=x, y=y, z=z)


def generate_line(
    sequence: Sequence, id="generate_strand"
) -> Tuple[Strand, list[Nucleotide]]:
    nucleotides: list[Nucleotide] = [
        Nucleotide(index=i, type=nt, coordinate=Coordinate(x=0, y=0, z=i * 6.5))
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

# 4. Generate Strand - Crude Simulation
def generate_strand(sequence: Sequence) -> Strand:
    strand, nucleotides = generate_line(sequence)
    nucleotides = crude_simulate(5, nucleotides, k=2, steps=200)
    strand = nucleotides_to_strand(nucleotides)
    return strand


# 5. Nucleotides to Strand
def nucleotides_to_strand(
    nucleotides: list[Nucleotide], id="generated_strand"
) -> Strand:
    strand = Strand(
        ID=[id + f"_{nt.index}" for nt in nucleotides],
        resname=[nt.type for nt in nucleotides],
        resid=[nt.index for nt in nucleotides],
        x_1=[nt.coordinate.x for nt in nucleotides],
        y_1=[nt.coordinate.y for nt in nucleotides],
        z_1=[nt.coordinate.z for nt in nucleotides],
    )
    return strand


if __name__ == "__main__":
    generator_tests_path = "data/generator_tests"
    # labels_path = "data/train_labels.csv"
    # sequences_path = "data/train_sequences.csv"
    labels_path = "data/validation_labels.csv"
    sequences_path = "data/validation_sequences.csv"
    labels = Labels(df=pd.read_csv(labels_path))
    sequences = Sequences(df=pd.read_csv(sequences_path))
    
    sequence = grab_random_sequence(sequences)
    pdb_id = sequence.target_id
    generated_strand = generate_strand(sequence)
    reference_strand = grab_strand(pdb_id, labels)
    generated_pdb = strand_to_pdb(generated_strand)
    reference_pdb = strand_to_pdb(reference_strand)
    # save with pdb_id
    save_path_generator = f"{generator_tests_path}/{pdb_id}_generated.pdb"
    save_path_reference = f"{generator_tests_path}/{pdb_id}_reference.pdb"
    save_pdb(generated_pdb, save_path_generator)
    save_pdb(reference_pdb, save_path_reference)
    # compute similarity
    compute_similarity(save_path_generator, save_path_reference)
    print("Completed Generation and Comparison for", pdb_id)