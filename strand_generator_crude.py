from tools import *

from pydantic import BaseModel
from numpy import ndarray
import numpy as np
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

