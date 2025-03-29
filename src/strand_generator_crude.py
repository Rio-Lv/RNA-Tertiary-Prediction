from tools import *

from pydantic import BaseModel
from numpy import ndarray
import numpy as np
# Crude Method
# Lets Take 5 nearest Nucleotides then apply displacement to single nucleotide

# 2. Displace Nucleotide with some formula
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


def displace_nucleotide(
    nucleotide: Nucleotide, neighbors: list[Nucleotide], distance_matrix: ndarray, k=1
):
    for neighbor in neighbors:
        orbital_pull(nucleotide, neighbor, distance_matrix, k=k)


def crude_simulate(N: int, nucleotides: list[Nucleotide], k=1, steps=50):
    print(" Number of Nucleotides: ", len(nucleotides))
    print(" Number of Steps: ", steps)
    print(" Number of Neighbors: ", N)
    for i in range(steps):
        print(f"Step: {i} / {steps}")
        distance_matrix = calculate_distance_matrix(nucleotides)
        for nt in nucleotides:
            neighbors = get_nearest_nucleotides(N, nt.index, nucleotides, distance_matrix)
            displace_nucleotide(nt, neighbors, distance_matrix, k=k)
        contract_nucleotides(nucleotides)
    return nucleotides

if __name__ == "__main__":
    # set dir to file location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("============ Strand Generator ===========")
    generator_tests_path = "data/generator_tests"
    labels_path = "data/train_labels.csv"
    sequences_path = "data/train_sequences.csv"
    # labels_path = "data/validation_labels.csv"
    # sequences_path = "data/validation_sequences.csv"
    labels = Labels(df=pd.read_csv(labels_path))
    sequences = Sequences(df=pd.read_csv(sequences_path))
    
    # sequence = grab_random_sequence(sequences)
    sequence = grab_sequence("9G7C_A", sequences)
    pdb_id = sequence.target_id
    nucleotides = sequence_to_nucleotide_line(sequence)
    nucleotides = crude_simulate(5, nucleotides, k=2, steps=2000)
    strand = nucleotides_to_strand(nucleotides)
    reference_strand = grab_strand(pdb_id, labels)
    generated_pdb = strand_to_pdb(strand)
    reference_pdb = strand_to_pdb(reference_strand)
    # save with pdb_id
    save_path_generator = f"{generator_tests_path}/{pdb_id}_generated.pdb"
    save_path_reference = f"{generator_tests_path}/{pdb_id}_reference.pdb"
    save_pdb(generated_pdb, save_path_generator)
    save_pdb(reference_pdb, save_path_reference)
    # compute similarity
    compute_similarity(save_path_generator, save_path_reference)
    print("Completed Generation and Comparison for", pdb_id)