import cmath
import math
import numpy as np
from tools import *
import torch
from torch import Tensor
import os
import random
import copy

EPS = 1e-8
K = 1  # Coupling constant
MAX_DELTA = 0.01  # Maximum delta for noise

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

    # # Move With Noise
    # for i in range(N):
    #     particles[i].phase += phase_shift[i]
    #     noise_mag = phase_shift[i] - natural_freq[i]
    #     # create unit vector in random direction in a sphere
    #     theta = np.random.uniform(0, 2 * np.pi)
    #     phi = np.random.uniform(0, np.pi)
    #     dx = noise_mag * np.sin(phi) * np.cos(theta)
    #     dy = noise_mag * np.sin(phi) * np.sin(theta)
    #     dz = noise_mag * np.cos(phi)
    #     particles[i].coord.x += dx
    #     particles[i].coord.y += dy
    #     particles[i].coord.z += dz

    # Move using phase wave gradient
    # 1. every particle emits a wave
    # 2. at every other particle if the other is on a slope of the wave towards the own particle
    # 3. move towards the other particle otherwise move away
    # 4. the wave is a sine wave but shifted of own phase which affect where on slope other particle is
    # 5. get gradient of wave a d = distance matrix [i][j]
    deltas = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if i != j:
                d = dist_matrix[i][j]
                slope = math.cos(d/particles[i].natural_freq + particles[i].phase)
                # apply sigmoid to magnitude of slope
                mag = -slope
                ux = (particles[i].coord.x - particles[j].coord.x) / d
                uy = (particles[i].coord.y - particles[j].coord.y) / d
                uz = (particles[i].coord.z - particles[j].coord.z) / d
                dx = ux * mag
                dy = uy * mag
                dz = uz * mag
                deltas[i][0] += dx
                deltas[i][1] += dy
                deltas[i][2] += dz
      
    # 1. Get Magnitude of deltas
    # 2. Use min delta and MAX_DELTA
    # 3. Re apply to particles          
    for i in range(N):
        mag = np.linalg.norm(deltas[i])
        ux = deltas[i][0] / mag
        uy = deltas[i][1] / mag
        uz = deltas[i][2] / mag
        new_mag = min(mag, MAX_DELTA)
        dx = ux * new_mag
        dy = uy * new_mag
        dz = uz * new_mag
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
    random.seed(42)
    return random.uniform(0, 2 * np.pi)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Example usage
    w = 0.1

    # particles = [
    #     Particle(Vector(0, 0, 0.5), natural_freq=w, phase=rand_phase()),
    #     Particle(Vector(1, 0.1, 0), natural_freq=w*2, phase=rand_phase()),
    #     Particle(Vector(2, -0.1, 0), natural_freq=w, phase=rand_phase()),
    #     Particle(Vector(3, 0, -1), natural_freq=w, phase=rand_phase()),
    #     Particle(Vector(4, 2, 0), natural_freq=w*2.1, phase=rand_phase()),
    #     Particle(Vector(5, 0, 1), natural_freq=w, phase=rand_phase()),
    # ]

    particles = []
    grid_size = 5
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                x = i + random.uniform(-0.1, 0.1)
                y = j + random.uniform(-0.1, 0.1)
                z = k + random.uniform(-0.1, 0.1)
                particles.append(
                    Particle(Vector(x,y,z), natural_freq=5, phase=rand_phase())
                )

    init_particles = particles.copy()

    dummy_seq_str = "A" * len(particles)
    coords = [p.coord for p in init_particles]
    initial_coords = copy.deepcopy(coords)
    recording = sync_particles(particles, n_iter=500)
    create_video(
        target_coords=copy.deepcopy(initial_coords),
        recording=recording,
        seq_str=dummy_seq_str,
        speed=10,
        save_path="videos/rna_waves.mp4",
    )
    coords = [p.coord for p in init_particles]
    plot_coords_list(dummy_seq_str, coords, initial_coords)
