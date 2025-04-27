import cmath
import math
import numpy as np
from tools import *
import torch
from torch import Tensor
import os
import random

EPS = 1e-8
K = 0.1  # Coupling constant


class Particle:
    def __init__(
        self,
        coord: Vector,  # Vector contains x, y, z
        natural_freq: float = 1,  # Angstrom per iteration
        phase: float = 0,
    ):
        self.coord = coord
        self.natural_freq = natural_freq
        self.phase = phase


def shift_phases(particles: list[Particle]) -> list[Particle]:
    coords = [p.coord for p in particles]
    coords = coord_list_to_matrix(coords)
    dist_matrix = coord_to_distance_matrix(coords).cpu() + EPS
    dist_matrix = dist_matrix.numpy()
    coupling = K / (dist_matrix**2)
    coupling[np.isinf(coupling)] = 0
    natural_freq = np.array([p.natural_freq for p in particles])
    phase = np.array([p.phase for p in particles])
    N = len(particles)
    phase_shift = np.zeros_like(natural_freq)
    for i in range(N):
        phase_shift[i] = natural_freq[i] 
        for j in range(N):
            if i != j:
                phase_shift[i] += (1 / N) * coupling[i][j] * np.sin(phase[j] - phase[i])

    for i in range(N):
        particles[i].phase += phase_shift[i]
        noise_mag = phase_shift[i] - natural_freq[i]
        # create unit vector in random direction in a sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        dx = noise_mag * np.sin(phi) * np.cos(theta)
        dy = noise_mag * np.sin(phi) * np.sin(theta)
        dz = noise_mag * np.cos(phi)
        particles[i].coord.x += dx
        particles[i].coord.y += dy
        particles[i].coord.z += dz

    return particles, coords


def sync_particles(particles: list[Particle], n_iter: int) -> list[Particle]:
    recording = []
    for iter in range(n_iter):
        print(f"Iteration {iter}/{n_iter}")
        particles, coords = shift_phases(particles)
        recording.append(coords)
    return recording


def rand_phase():
    return random.uniform(0, 2 * np.pi)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Example usage
    w = 0.1

    particles = [
        Particle(Vector(0, 0, 0), natural_freq=w, phase=rand_phase()),
        Particle(Vector(1, 0, 0), natural_freq=w, phase=rand_phase()),
        Particle(Vector(1, 1, 0), natural_freq=w, phase=rand_phase()),
        Particle(Vector(1, 1, 1), natural_freq=w, phase=rand_phase()),
        Particle(Vector(2, 1, 1), natural_freq=w, phase=rand_phase()),
        Particle(Vector(3, 1, 1), natural_freq=w, phase=rand_phase()),
    ]

    initial_coords = [p.coord for p in particles.copy()]

    recording = sync_particles(particles, n_iter=500)
    dummy_seq_str = "A" * len(particles)
    create_video(
        target_coords=initial_coords,
        recording=recording,
        seq_str=dummy_seq_str,
        speed=5,
        save_path="videos/rna_waves.mp4",
    )
