import cmath
import math
import numpy as np
from tools import Vector, coord_list_to_matrix, coord_to_distance_matrix
import torch 
from torch import Tensor
EPS = 1e-8

class Particle:
    def __init__(
        self,
        coord: Vector, # Vector contains x, y, z
        natural_freq: float = 1, # Angstrom per iteration
        phase: float = 0,
    ):
        self.coord = coord 
        self.natural_freq = natural_freq
        self.phase = phase 


def calc_phase_shift(particles: list[Particle]) -> float:
    coords = [p.coord for p in particles]
    coords = coord_list_to_matrix(coords)
    dist_matrix = coord_to_distance_matrix(coords).cpu() + EPS
    dist_matrix = dist_matrix.numpy() 
    coupling = 1/dist_matrix
    coupling[np.isinf(coupling)] = 0
    natural_freq = np.array([p.natural_freq for p in particles])
    phase = np.array([p.phase for p in particles])
    N = len(particles)
    phase_shift = np.zeros_like(natural_freq)
    for i in range(N):
        phase_shift[i] = natural_freq[i] 
        for j in range(N):
            if i != j:
                phase_shift[i] += (1/N) * coupling[i][j] * np.sin(phase[j] - phase[i])
    return phase_shift


if __name__ == "__main__":
    # Example usage
    particles = [
        Particle(Vector(0, 0, 0), natural_freq=1, phase=0),
        Particle(Vector(1, 0, 0), natural_freq=2.1, phase=1.2),
        Particle(Vector(0, 1, 0), natural_freq=1, phase=0),
        Particle(Vector(1, 1, 0), natural_freq=1, phase=1),
        Particle(Vector(1, 1, 1), natural_freq=1, phase=0),
        Particle(Vector(1, 1, -1), natural_freq=2, phase=1),
    ]

    phase_shift = calc_phase_shift(particles)
    print("Phase shift:", phase_shift)
