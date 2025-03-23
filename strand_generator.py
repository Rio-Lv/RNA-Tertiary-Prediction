from tools import *

# import tensorflow as tf
from typing import Literal, Tuple
from pydantic import BaseModel
import random
import numpy as np
from numpy import ndarray


class Coordinate(BaseModel):
    x: float
    y: float
    z: float


class Nucleotide(BaseModel):
    index: int
    type: Literal["A", "C", "G", "U"]
    coordinate: Coordinate = Coordinate(x=0, y=0, z=0)


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


# Crude Method
# Lets Take 5 nearest Nucleotides then apply displacement to single nucleotide
# 1. Calculate Distance Matrix
def calculate_distance_matrix(nucleotides: list[Nucleotide]) -> ndarray:
    matrix = []
    for nt in nucleotides:
        distances = []
        for nt2 in nucleotides:
            distance = (
                (nt.coordinate.x - nt2.coordinate.x) ** 2
                + (nt.coordinate.y - nt2.coordinate.y) ** 2
                + (nt.coordinate.z - nt2.coordinate.z) ** 2
            ) ** 0.5
            distances.append(distance)
        matrix.append(distances)
    return np.array(matrix)


# 2. Get 5 nearest Nucleotides
def get_nearest_nucleotides(
    N: int, nucleotide: Nucleotide, nucleotides: list[Nucleotide], distance_matrix
) -> list[Nucleotide]:
    distances = distance_matrix[nucleotide.index]
    nearest = np.argsort(distances)
    return [nucleotides[i] for i in nearest[1:N]]


# 3. Displace Nucleotide with some formula
class OrbitalRadii(BaseModel):
    radii: dict[str, float] = {
        "AU": 6.5,
        "UA": 6.5,
        "CG": 6.5,
        "GC": 6.5,
    }

def orbital_pull(
    nucleotide_1: Nucleotide,
    nucleotide_2: Nucleotide,
    distance_matrix: ndarray,
    k=1,
    noise=0.01,
):
    type_1 = nucleotide_1.type
    type_2 = nucleotide_2.type
    orbital_type = str(type_1 + type_2)
    radii_dict = OrbitalRadii().radii
    radii = radii_dict.get(orbital_type, None)
    if radii is None:
        return

    distance = distance_matrix[nucleotide_1.index][nucleotide_2.index]
    if distance < radii:
        k = -k

    force = k * min(1 / max(abs(distance - radii),0.01) ** 2, 0.4)

    ux = (nucleotide_2.coordinate.x - nucleotide_1.coordinate.x) / distance
    uy = (nucleotide_2.coordinate.y - nucleotide_1.coordinate.y) / distance
    uz = (nucleotide_2.coordinate.z - nucleotide_1.coordinate.z) / distance
    nucleotide_1.coordinate.x += force * ux + random.uniform(-noise, noise)
    nucleotide_1.coordinate.y += force * uy + random.uniform(-noise, noise)
    nucleotide_1.coordinate.z += force * uz + random.uniform(-noise, noise)


def correct_bonds(nucleotides: list[Nucleotide]):
    for i in range(1, len(nucleotides)):
        nt1 = nucleotides[i - 1]
        nt2 = nucleotides[i]
        distance = (
            (nt1.coordinate.x - nt2.coordinate.x) ** 2
            + (nt1.coordinate.y - nt2.coordinate.y) ** 2
            + (nt1.coordinate.z - nt2.coordinate.z) ** 2
        ) ** 0.5
        d = distance - 6.5
        ux = (nt2.coordinate.x - nt1.coordinate.x) / distance
        uy = (nt2.coordinate.y - nt1.coordinate.y) / distance
        uz = (nt2.coordinate.z - nt1.coordinate.z) / distance
        dx = ux * d
        dy = uy * d
        dz = uz * d
        for nt in nucleotides[i:]:
            nt.coordinate.x -= dx
            nt.coordinate.y -= dy
            nt.coordinate.z -= dz


def displace_nucleotide(
    nucleotide: Nucleotide, neighbors: list[Nucleotide], distance_matrix: ndarray, k=1
):
    for neighbor in neighbors:
        orbital_pull(nucleotide, neighbor, distance_matrix, k=k)


def crude_simulate(N: int, nucleotides: list[Nucleotide], k=1, steps=100):
    for i in range(steps):
        print(f"Step: {i} / {steps}")
        distance_matrix = calculate_distance_matrix(nucleotides)
        for nt in nucleotides:
            neighbors = get_nearest_nucleotides(N, nt, nucleotides, distance_matrix)
            displace_nucleotide(nt, neighbors, distance_matrix, k=k)
        correct_bonds(nucleotides)
    return nucleotides

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
    labels_path = "data/train_labels.csv"
    sequences_path = "data/train_sequences.csv"
    # labels_path = "data/validation_labels.csv"
    # sequences_path = "data/validation_sequences.csv"
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