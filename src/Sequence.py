import os
from typing import Optional
from torch import Tensor
from torch import nn
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import copy
import math
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
from _deprecated_tools import compute_similarity
from torch.utils.data import DataLoader, TensorDataset, random_split

from torch.optim import Adam
from tools import *

# set here to cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ====== CONSTANTS ======
SEQUENCE_SIZE = 300

N_SEQUENCES = 50
# N_NEAREST_NEIGBORS = 30  # If using n nearest neighbors for adjustment
# MAX_DISTANCE = 32  # If using neightbor within distance for adjustment
# USE_NEIGHBORS = False  # If using n nearest neighbors for adjustment

ITERATIONS = 5000
ITERATIONS_PER_RESIDUE = (ITERATIONS//SEQUENCE_SIZE)//2 
TEMPERATURE = 0.1
MAX_DELTA = 0.5

LABELS_PATH = "data/train_labels.csv"
SEQUENCES_PATH = "data/train_sequences.csv"
SEQUENCE_INDEX = 868

MAX_SPINE_SPACE = 5  # Maximum distance between two points in the spine

OPEN_PLOT = True  # If True, will open a plot window for each sequence
GRAVITY = 0.001
VIDEO_SPEED = ITERATIONS // 200  # Speed of the video in frames per second

DIR_BIAS_X = 1  # Bias for the x direction in random walk
DELTA_DROP_RATE = 0.95  # Rate at which deltas are dropped
MAX_INDEX_DIFF = SEQUENCE_SIZE
# NOISY_SOURCE_MATRIX = True


# ====== SEQUENCE ======
class Sequence:
    """
    Contains a type encodeding of a sequence.
    """

    seq_str: str
    seq_id: Optional[str]
    coords: Optional[list[Vector]]
    source_coords: Optional[list[Vector]]
    distance_matrix: Tensor
    encoding: Tensor
    coord_matrix: Tensor

    def __init__(self, seq_str: str, seq_id: str = None, coords: list[Vector] = None):
        self.seq_str = seq_str
        self.seq_id = seq_id
        self.encoding = encode_str(seq_str)
        self.coords = (
            copy.deepcopy(coords) if coords else self._coords_to_noise(walk=True)
        )
        self.source_coords = copy.deepcopy(self.coords)
        self.coord_matrix = coords_list_to_matrix(self.coords)
        self.distance_matrix = coord_to_distance_matrix(self.coord_matrix)

    def __repr__(self):
        # Print Seq String but if longer than 50 chars use ellipsis
        repr_seq_str = (
            self.seq_str[:50] + "..." if len(self.seq_str) > 50 else self.seq_str
        )

        msg = "\n"
        msg += " ========= SEQUENCE ======== \n"
        msg += "\n"
        msg += f"ID: {self.seq_id} \n"
        msg += f"SEQ: {repr_seq_str} \n"
        msg += "---------------------------- \n"
        msg += "ENCODING: \n"
        msg += f"{self.encoding} \n"
        msg += "---------------------------- \n"
        msg += "\n"
        msg += "DISTANCE MATRIX: \n"
        msg += f"{self.distance_matrix} \n"
        msg += "---------------------------- \n"
        return msg

    def subset(self, start: int, end: int):
        """
        Grab a subset of the sequence.
        :param start: Start index
        :param end: End index
        :return: Subset of the sequence
        """
        seq_id = self.seq_id
        seq_str = self.seq_str[start:end]
        coords = self.coords[start:end]
        return Sequence(seq_str=seq_str, seq_id=seq_id, coords=coords)

    @staticmethod
    def to_pdb(coords: list[Vector], seq_str: str, save_path: str = None) -> str:
        pdb_str = ""
        for i in range(len(coords)):
            x = coords[i].x
            y = coords[i].y
            z = coords[i].z
            resname = seq_str[i]
            pdb_str += f"ATOM  {i+1:5d}  CA   {resname} A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"

        if save_path:
            with open(save_path, "w") as f:
                f.write(pdb_str)
            print(f"Saved PDB to {save_path}")
        return pdb_str

    @staticmethod
    def compute_similarity_us_align(
        gen_path="seq_output/seq_generated.pdb", target_path="seq_output/seq_target.pdb"
    ):

        compute_similarity(gen_path, target_path)

    # ====== TESTING (Sequence Class) ======
    def _coords_to_noise(self, walk: bool = True, noise_k: float = 5):
        """
        Testing function, replace coords with random noise

        Generate a 3D random walk.
        First coordinate is random, subsequent are 5.5A in a random direction.
        """
        if walk:
            n = len(self.seq_str)

            r = 5.5
            dir_bias_x = DIR_BIAS_X
            coords = []
            curr_pos = Vector(
                random.uniform(-r, r),
                random.uniform(-r / dir_bias_x, r / dir_bias_x),
                random.uniform(-r / dir_bias_x, r / dir_bias_x),
            )
            coords.append(curr_pos)

            for _ in range(1, n):
                theta = random.uniform(0, 2 * math.pi)
                phi = random.uniform(0, math.pi)
                dx = r * math.sin(phi) * math.cos(theta)
                dy = r * math.sin(phi) * math.sin(theta)
                dz = r * math.cos(phi)
                curr_pos = Vector(curr_pos.x + dx, curr_pos.y + dy, curr_pos.z + dz)
                coords.append(curr_pos)
            self.coords = coords
            return coords
        else:
            noise = noise_k
            for i in range(len(self.coords)):
                self.coords[i].x = random.uniform(-noise, noise)
                self.coords[i].y = random.uniform(-noise, noise)
                self.coords[i].z = random.uniform(-noise, noise)
            return self.coords

    def _test_adjust_coords_video(
        self,
        iterations: int,
        speed: int,
        video_filename="seq_output/adjustment.mp4",
        interval=33,
    ):
        """
        Adjust the coordinates to match the distance matrix and output a video
        showing the adjustment process over multiple iterations.

        The original coordinates are shown in gray while the adjusted coordinates
        are plotted in red.

        Parameters:
        video_filename (str): The filename of the output video.
        iterations (int): Total number of adjustment iterations.
        speed (int): Render every Nth frame (downsampling factor for recording).
        interval (int): Delay between frames in milliseconds.
        """
        print("Starting video generation...")
        start_time = time.time()
        self._coords_to_noise()

        # Adjust the coordinates and record the intermediate states.
        coords_adjusted, recording = adjust_coords(
            n_iter=iterations,
            coords=self.coords,
            target_matrix=self.distance_matrix,
            temperature=TEMPERATURE,
            delta_drop_rate=DELTA_DROP_RATE,
            max_delta=MAX_DELTA,
            iterations_per_residue=ITERATIONS_PER_RESIDUE,
            max_index_diff=MAX_INDEX_DIFF,
        )

        gen_path = "seq_output/seq_generated.pdb"
        target_path = "seq_output/seq_target.pdb"

        self.to_pdb(
            coords_adjusted,
            self.seq_str,
            save_path=gen_path,
        )
        self.to_pdb(
            self.source_coords,
            self.seq_str,
            save_path=target_path,
        )

        self.compute_similarity_us_align(
            gen_path=gen_path,
            target_path=target_path,
        )

        # Store the original coordinates (as a list of Vectors).
        original_coords: list[Vector] = self.source_coords

        # Convert each recorded tensor (shape: [N, 3]) into a list of Vector objects.
        recording_coords: list[list[Vector]] = []
        for frame_tensor in recording:
            frame_coords = [
                Vector(coord[0].item(), coord[1].item(), coord[2].item())
                for coord in frame_tensor
            ]
            recording_coords.append(frame_coords)

        # Downsample the recording so that only every Nth frame is rendered.
        recording_coords = recording_coords[::speed]
        num_frames = len(recording_coords)

        # Compute bounding box limits based on the original coordinates with 25% padding.
        x_orig_vals = [coord.x for coord in original_coords]
        y_orig_vals = [coord.y for coord in original_coords]
        z_orig_vals = [coord.z for coord in original_coords]

        x_min, x_max = min(x_orig_vals), max(x_orig_vals)
        y_min, y_max = min(y_orig_vals), max(y_orig_vals)
        z_min, z_max = min(z_orig_vals), max(z_orig_vals)

        # Ensure nonzero ranges.
        x_range = x_max - x_min if (x_max - x_min) != 0 else 1.0
        y_range = y_max - y_min if (y_max - y_min) != 0 else 1.0
        z_range = z_max - z_min if (z_max - z_min) != 0 else 1.0

        x_pad = 0.25 * x_range
        y_pad = 0.25 * y_range
        z_pad = 0.25 * z_range

        x_lim = (x_min - x_pad, x_max + x_pad)
        y_lim = (y_min - y_pad, y_max + y_pad)
        z_lim = (z_min - z_pad, z_max + z_pad)

        # Create a 3D plot for the animation.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        def init():
            ax.clear()
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            # Plot original coordinates.
            x_orig = [coord.x for coord in original_coords]
            y_orig = [coord.y for coord in original_coords]
            z_orig = [coord.z for coord in original_coords]
            ax.scatter(x_orig, y_orig, z_orig, color="gray", s=100, label="Original")
            ax.plot(
                x_orig,
                y_orig,
                z_orig,
                color="gray",
                alpha=0.7,
                linewidth=2,
                label="Original Path",
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Adjustment Process")
            return []

        def update(frame):
            # Print progress every 10 frames.
            if (frame + 1) % 10 == 0 or frame == 0:
                print(f"Processing frame {frame+1}/{num_frames}")

            # Optionally, align the current recorded frame to the original coordinates.
            current_coords = align(recording_coords[frame], original_coords)
            x_adj = [coord.x for coord in current_coords]
            y_adj = [coord.y for coord in current_coords]
            z_adj = [coord.z for coord in current_coords]

            ax.clear()
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)

            # Plot the static original coordinates.
            x_orig = [coord.x for coord in original_coords]
            y_orig = [coord.y for coord in original_coords]
            z_orig = [coord.z for coord in original_coords]
            ax.scatter(x_orig, y_orig, z_orig, color="gray", s=100, label="Original")
            ax.plot(
                x_orig,
                y_orig,
                z_orig,
                color="gray",
                alpha=0.7,
                linewidth=2,
                label="Original Path",
            )
            # Plot the adjusted (dynamic) coordinates.
            ax.scatter(x_adj, y_adj, z_adj, color="red", s=100, label="Adjusted")
            ax.plot(
                x_adj,
                y_adj,
                z_adj,
                color="red",
                alpha=0.7,
                linewidth=2,
                label="Adjusted Path",
            )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Adjustment Frame {frame+1}")
            ax.legend(loc="upper right")
            return []

        # Create the animation object using the decimated recording.
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(num_frames),
            init_func=init,
            interval=interval,
            repeat=False,
        )

        # Save the video using the FFmpeg writer.
        Writer = animation.writers["ffmpeg"]
        writer = Writer(
            fps=1000 // interval, metadata=dict(artist="Your Name"), bitrate=1800
        )
        ani.save(video_filename, writer=writer)
        plt.close(fig)

        elapsed_time = time.time() - start_time
        print(f"Video saved to {video_filename} in {elapsed_time:.2f} seconds")

        if OPEN_PLOT:
            plot_coords_list([original_coords, self.coords], ["Original", "Adjusted"])


# ====== DATA PPEPERATION ======
class SequenceDataset:
    """
    Inputs is the Sequence Encoding
    Output is the distance matrix
    """

    source_sequences: list[Sequence]

    def __init__(self, n_sequences: int, sequence_size: int):
        self.n_sequences = n_sequences
        self.sequence_size = sequence_size
        self.source_sequences = self.generate_source_sequences()

    def generate_source_sequences(self):
        sequence_size = self.sequence_size
        label_df = pd.read_csv(LABELS_PATH)
        sequences_df = pd.read_csv(SEQUENCES_PATH)
        # using a windowed approach
        sequences = []
        n_sequences = self.n_sequences

        for i in range(len(sequences_df)):

            if len(sequences) >= n_sequences:
                print("Target number of sequences reached.")
                break
            seq_id = sequences_df.iloc[i]["target_id"]
            seq_str = sequences_df.iloc[i]["sequence"]

            if len(seq_str) < sequence_size:
                continue

            for i in range(len(seq_str) - sequence_size + 1):
                start_index = i
                end_index = start_index + sequence_size

                if len(sequences) >= n_sequences:
                    print("Target number of sequences reached.")
                    break

                print(
                    f"Processing sequence {len(sequences)}/{n_sequences} ({seq_id}) FROM INDEX {start_index} to {end_index})"
                )

                # seq labels is label df where seq_id is included ID col
                seq_labels = label_df[label_df["ID"].str.contains(seq_id, na=False)]

                x_1 = seq_labels["x_1"].tolist()
                y_1 = seq_labels["y_1"].tolist()
                z_1 = seq_labels["z_1"].tolist()

                coords = []
                for x, y, z in zip(x_1, y_1, z_1):
                    coords.append(Vector(x, y, z))

                seq = Sequence(
                    seq_str=seq_str[start_index:end_index],
                    seq_id=f"{seq_id} split_{i+1}",
                    coords=coords[start_index:end_index],
                )

                sequences.append(seq)
        self.source_sequences = sequences
        return sequences

    def get_random_sequence(self):
        random_index = random.randint(0, len(self.source_sequences) - 1)
        seq = self.source_sequences[random_index]
        curr_try = 0
        max_tries = 2000
        while len(seq.coords) < SEQUENCE_SIZE:
            random_index = random.randint(0, len(self.source_sequences) - 1)
            seq = self.source_sequences[random_index]
            curr_try += 1
            if curr_try > max_tries:
                print("Max tries reached, returning random sequence.")
                break
        seq = seq.subset(0, SEQUENCE_SIZE)
        print(f"Random Sequence: {random_index}")
        return seq

    def get_stats(self):
        """
        Get the stats of the dataset.
        1. Number of sequences
        2. Average str length of sequence in 'sequence' col
        3. Min length
        4. Max length
        5. Create a histogram of the lengths
        """
        sequences = pd.read_csv(SEQUENCES_PATH)
        print(f"Number of sequences: {len(sequences)}")
        print(f"Average sequence length: {sequences['sequence'].str.len().mean()}")
        print(f"Min sequence length: {sequences['sequence'].str.len().min()}")
        print(f"Max sequence length: {sequences['sequence'].str.len().max()}")
        print(f"Standard deviation: {sequences['sequence'].str.len().std()}")

        # Plot histogram of sequence lengths
        # limit the x-axis to 300
        plt.hist(sequences["sequence"].str.len(), bins=300)
        plt.xlim(0, 1000)
        plt.xlabel("Sequence Length")
        plt.ylabel("Frequency")
        plt.title("Histogram of Sequence Lengths")
        plt.show()


# ======== MODELS ==========

if __name__ == "__main__":

    print("Starting Sequence Class Test")

    # Test Sequence from Seq String
    # seq = Sequence("ACGTAACGUUU")
    # seq.test_adjust_coords()
    # seq.test_adjust_coords_video()
    # print(seq)

    # # ============ Test 1 ==============
    # # Test the Sequence Dataset class
    # seq_dataset = SequenceDataset()
    # print(len(seq_dataset.source_sequences))
    # # Initialize a real sequence (Distance Matrix Assigned)
    # seq = seq_dataset.get_random_sequence()
    # # seq.test_adjust_coords_video()
    # # 1. Replace Coordinate with Random Noise
    # seq._coords_to_noise()
    # # 2. Adjust Coordinates to match the distance matrix
    # seq.adjust_coords(n_iter=100)
    # # 3. Align the sequence to a target sequence
    # seq.align(seq.source_coords)
    # # 4. Save the adjusted coordinates to a PDB file
    # seq.to_pdb("data/pdbs_fake/sequence_class_test.pdb")
    # # 5. Plot the original and adjusted coordinates (optional)
    # seq.plot([seq.source_coords, seq.coords], ["Original", "Adjusted"])

    # =============== Test 2 ==============
    print("Loading Sequence Dataset")
    seq_dataset = SequenceDataset(n_sequences=N_SEQUENCES, sequence_size=SEQUENCE_SIZE)
    print(len(seq_dataset.source_sequences))
    # Initialize a real sequence (Distance Matrix Assigned)
    # seq_dataset.get_stats()
    print("Loading Random Sequence")
    # seq = seq_dataset.get_random_sequence()
    seq = seq_dataset.source_sequences[0]
    print(f"Sequence Length: {len(seq.seq_str)}")
    # seq._coords_to_noise()
    # seq.adjust_coords(n_iter=100, use_neighbors=USE_NEIGHBORS)
    seq._test_adjust_coords_video(iterations=ITERATIONS, speed=VIDEO_SPEED)

    # Sequence.compute_similarity_us_align()
