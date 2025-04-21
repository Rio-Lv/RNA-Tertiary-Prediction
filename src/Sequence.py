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

from torch.utils.data import DataLoader, TensorDataset, random_split

from torch.optim import Adam
from tools import *
from SpineModel import SpineModel

# set here to cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ====== CONSTANTS ======
SUBSET_LEN = 77

N_SEQUENCES = 50
# N_NEAREST_NEIGBORS = 30  # If using n nearest neighbors for adjustment
# MAX_DISTANCE = 32  # If using neightbor within distance for adjustment
# USE_NEIGHBORS = False  # If using n nearest neighbors for adjustment

ITERATIONS = 6000

ITERATIONS_PER_RESIDUE = 4
ITERATIONS_PER_RESIDUE_SPINE = 5

MAX_DELTA = 0.5
MAX_DELTA_SPINE = 0.5

TEMPERATURE = 0.1

LABELS_PATH = "data/train_labels.csv"
SEQUENCES_PATH = "data/train_sequences.csv"

SEQUENCE_INDEX = 868

# MAX_SPINE_SPACE = 5  # Maximum distance between two points in the spine

OPEN_PLOT = True  # If True, will open a plot window for each sequence
GRAVITY = 0.001

VIDEO_SPEED = ITERATIONS // 200  # Speed of the video in frames per second

DIR_BIAS_X = 1  # Bias for the x direction in random walk
ACTIVE_KEEP_RATE = 0.1  # Rate at which deltas are dropped
MAX_INDEX_DIFF = 200
# NOISY_SOURCE_MATRIX = True

SCORES_MIN_SEQ_LEN = 50
SCORES_MAX_SEQ_LEN = 100


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

    def _primary_test(self):
        target_coords = self.coords.copy()
        input_coords = self._coords_to_noise()

        coords_adjusted, recording = adjust_coords(
            n_iter=ITERATIONS,
            input_coords=input_coords,
            target_matrix=self.distance_matrix,
            temperature=TEMPERATURE,
            active_keep_rate=ACTIVE_KEEP_RATE,
            max_delta=MAX_DELTA,
            iterations_per_residue=ITERATIONS_PER_RESIDUE,
            max_index_diff=MAX_INDEX_DIFF,
        )

        # Save the adjusted coordinates to a PDB file
        self.to_pdb(
            coords_adjusted,
            self.seq_str,
            save_path="seq_output/seq_generated.pdb",
        )
        # Save the original coordinates to a PDB file
        self.to_pdb(
            target_coords,
            self.seq_str,
            save_path="seq_output/seq_target.pdb",
        )
        self.compute_similarity_us_align(
            gen_path="seq_output/seq_generated.pdb",
            target_path="seq_output/seq_target.pdb",
        )
        create_video(
            target_coords=target_coords,
            recording=recording,
            seq_str=self.seq_str,
            speed=VIDEO_SPEED,
            save_path="seq_output/adjustment.mp4",
            interval=33,
        )

        plot_coords_list(
            seq_str=self.seq_str,
            coords=coords_adjusted,
            reference_coords=target_coords,
        )


# ====== DATA PPEPERATION ======
class SequenceDataset:
    """
    Inputs is the Sequence Encoding
    Output is the distance matrix
    """

    subset_sequences: list[Sequence]

    def __init__(self, n_sequences: int, subset_len: int = None):
        self.n_sequences = n_sequences
        self.labels = pd.read_csv(LABELS_PATH)
        self.sequences = pd.read_csv(SEQUENCES_PATH)

        self.subset_len = subset_len
        self.subset_sequences = self.generate_subset_sequences()

    def generate_subset_sequences(self):
        subset_len = self.subset_len
        label_df = self.labels
        sequences_df = self.sequences
        # using a windowed approach
        sequences = []
        n_sequences = self.n_sequences

        for i in range(len(sequences_df)):

            if len(sequences) >= n_sequences:
                print("Target number of sequences reached.")
                break
            seq_id = sequences_df.iloc[i]["target_id"]
            seq_str = sequences_df.iloc[i]["sequence"]

            if len(seq_str) < subset_len:
                continue

            for i in range(len(seq_str) - subset_len + 1):
                start_index = i
                end_index = start_index + subset_len

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
        self.subset_sequences = sequences
        return sequences

    def get_random_subset(self, retries:int = 0):
        if retries > 100:
            return print("retries get random subset exceeded 100")
        random_index = random.randint(0, len(self.subset_sequences) - 1)
        seq = self.subset_sequences[random_index]
        curr_try = 0
        max_tries = 2000
        while len(seq.coords) < SUBSET_LEN:
            random_index = random.randint(0, len(self.subset_sequences) - 1)
            seq = self.subset_sequences[random_index]
            
            curr_try += 1
            if curr_try > max_tries:
                print("Max tries reached, returning random sequence.")
                break
        seq = seq.subset(0, SUBSET_LEN)
        # ensure seq does not contain nan in vectors
        for i, coord in enumerate(seq.coords):
            if math.isnan(coord.x) or math.isnan(coord.y) or math.isnan(coord.z):
                return self.get_random_subset(retries=retries+1)
        
        print(f"Random Sequence: {random_index}")
        return seq

    def get_sequence_ids(
        self, min_len: int = SCORES_MAX_SEQ_LEN, max_len: int = SCORES_MAX_SEQ_LEN
    ) -> list[str]:
        """
        Get the sequence IDs from the dataset.
        But only where sequence (str) is within length limits.
        :return: list of sequence IDs
        """
        sequence_ids = []
        for i in range(len(self.sequences)):
            seq_str = self.sequences.iloc[i]["sequence"]
            if len(seq_str) < min_len or len(seq_str) > max_len:
                continue
            seq_id = self.sequences.iloc[i]["target_id"]
            sequence_ids.append(seq_id)
        return sequence_ids

    def get_sequence(self, seq_id: str):
        labels = self.labels
        sequences = self.sequences
        seq_str = sequences[sequences["target_id"] == seq_id]["sequence"].values[0]
        seq_labels = labels[labels["ID"].str.contains(seq_id, na=False)]
        x_1 = seq_labels["x_1"].tolist()
        y_1 = seq_labels["y_1"].tolist()
        z_1 = seq_labels["z_1"].tolist()
        coords = []
        for x, y, z in zip(x_1, y_1, z_1):
            coords.append(Vector(x, y, z))
        seq = Sequence(
            seq_str=seq_str,
            seq_id=f"{seq_id}",
            coords=coords,
        )
        seq.coords = coords
        seq.source_coords = coords
        seq.coord_matrix = coords_list_to_matrix(seq.coords)
        seq.distance_matrix = coord_to_distance_matrix(seq.coord_matrix)
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


def analyse_scores(N: int):
    seq_sizes = []
    scores = []
    mirrored_scores = []
    for iter in range(N):


        # TODO: Work on grabbing full sequence instead of subset
        # seq_dataset = SequenceDataset(n_sequences=1, subset_len=SCORES_MIN_SEQ_LEN)
        # random_seq_id = random.choice(
        #     seq_dataset.get_sequence_ids(
        #         min_len=SCORES_MIN_SEQ_LEN, max_len=SCORES_MAX_SEQ_LEN
        #     )
        # )
        # seq = seq_dataset.get_sequence(seq_id=random_seq_id)
        
        
        seq_dataset = SequenceDataset(n_sequences=N_SEQUENCES, subset_len=SUBSET_LEN)
        # print(len(seq_dataset.subset_sequences))
        # Initialize a real sequence (Distance Matrix Assigned)
        seq = seq_dataset.get_random_subset()
        seq_size = len(seq.seq_str)

        target_matrix = seq.distance_matrix.clone()
        # print("Target Matrix: ", target_matrix)
        spine_model = SpineModel()
        spine_coords, _ = spine_model.construct_spine_coords(
            temperature=TEMPERATURE,
            iterations_per_residue=ITERATIONS_PER_RESIDUE_SPINE,
            seq_str=seq.seq_str,
            max_delta=MAX_DELTA_SPINE,
            target_matrix=target_matrix,
        )
        adjusted_coords = correct_chirality(spine_coords)
        target_pdb_path = "seq_output/sequence_source.pdb"
        generated_pdb_path = "seq_output/sequence_generatored.pdb"
        mirrored_pdb_path = "seq_output/sequence_mirrored.pdb"

        Sequence.to_pdb(coords=seq.coords, seq_str=seq.seq_str, save_path=target_pdb_path)
        Sequence.to_pdb(
            coords=adjusted_coords, seq_str=seq.seq_str, save_path=generated_pdb_path
        )
        Sequence.to_pdb(
            coords=mirror(adjusted_coords),
            seq_str=seq.seq_str,
            save_path=mirrored_pdb_path,
        )

        score = compute_similarity(path_1=generated_pdb_path, path_2=target_pdb_path)
        
        if score == 0:
            print("SPINE COORDS", spine_coords)
            print("ADJUSTED COORDS", adjusted_coords)
            break
        
        mirror_score = compute_similarity(path_1=mirrored_pdb_path, path_2=target_pdb_path)
        print(f"SCORE: {score} --   MIRROR SCORE: {mirror_score}")
        print(
            f"=== ANALYSED {iter+1} / {N} - Score {score} - Mirror Score {mirror_score} ==="
        )
        print(f"--- Sequence Size: {seq_size} ---")
        scores.append(score)
        mirrored_scores.append(mirror_score)
        seq_sizes.append(seq_size)

    # # sort both by seq_sizes
    # [print(score) for score in scores]
    # [print(seq_size) for seq_size in seq_sizes]

    seq_sizes, scores = zip(*sorted(zip(seq_sizes, scores), key=lambda x: x[0]))
    [
        print(
            f"Sequence Size: {seq_sizes[i]}, Score: {scores[i]}, Mirror Score: {mirrored_scores[i]}"
        )
        for i in range(len(seq_sizes))
    ]
    valid_scores = [s for s in scores if s is not None]
    # take make value from either scores or mirrored_scores

    max_scores = [max(scores[i], mirrored_scores[i]) for i in range(len(scores))]
    min_scores = [min(scores[i], mirrored_scores[i]) for i in range(len(scores))]
    if valid_scores:  # avoid ZeroDivisionError
        average = sum(valid_scores) / len(valid_scores)
        average_mirrored = sum(mirrored_scores) / len(mirrored_scores)
        avarage_max = sum(max_scores) / len(max_scores)
        average_min = sum(min_scores) / len(min_scores)
        print(f"Average TM0score: {average:.5f}")
        print(f"Average mirrored TM0score: {average_mirrored:.5f}")
        print(f"Average max score: {avarage_max:.5f}")
        print(f"Average min score: {average_min:.5f}")
    else:
        print("No valid scores were returned.")


def analyse_one():
    # Test the Sequence Dataset class
    seq_dataset = SequenceDataset(n_sequences=N_SEQUENCES, subset_len=SUBSET_LEN)
    # print(len(seq_dataset.subset_sequences))
    # Initialize a real sequence (Distance Matrix Assigned)
    seq = seq_dataset.get_random_subset()
    # seq.test_adjust_coords_video()
    # 1. Replace Coordinate with Random Noise
    # seq._coords_to_noise()
    # 1.2 Replace Coordinate with Contrsucted Spine Model

    target_matrix = seq.distance_matrix.clone()

    spine_model: SpineModel = SpineModel()
    spine_coords, spine_recording = spine_model.construct_spine_coords(
        seq_str=seq.seq_str, target_matrix=target_matrix
    )
    # spine_matrix = coord_to_distance_matrix(coords_list_to_matrix(spine_coords))
    # print("Spine Matrix: ", spine_matrix)

    # print("Target Matrix: ", target_matrix)

    # # swap target matrix with spine matrix whee abs(i-j) < 5
    # for i in range(len(target_matrix)):
    #     for j in range(len(target_matrix)):
    #         if abs(i - j) < 5:
    #             target_matrix[i][j] = spine_matrix[i][j]

    adjusted_coords = correct_chirality(spine_coords)
    recording = spine_recording
    # adjusted_coords, recording = adjust_coords(
    #     n_iter=ITERATIONS,
    #     input_coords=spine_coords,
    #     target_matrix=target_matrix,
    #     temperature=TEMPERATURE,
    #     # iterations_per_residue=ITERATIONS_PER_RESIDUE,
    #     active_keep_rate=ACTIVE_KEEP_RATE,
    #     max_delta=MAX_DELTA,
    # )

    target_pdb_path = "seq_output/sequence_source.pdb"
    generated_pdb_path = "seq_output/sequence_generatored.pdb"
    mirrored_pdb_path = "seq_output/sequence_mirrored.pdb"

    Sequence.to_pdb(coords=seq.coords, seq_str=seq.seq_str, save_path=target_pdb_path)
    Sequence.to_pdb(
        coords=adjusted_coords, seq_str=seq.seq_str, save_path=generated_pdb_path
    )
    Sequence.to_pdb(
        coords=mirror(adjusted_coords),
        seq_str=seq.seq_str,
        save_path=mirrored_pdb_path,
    )

    score = compute_similarity(path_1=generated_pdb_path, path_2=target_pdb_path)
    mirror_score = compute_similarity(path_1=mirrored_pdb_path, path_2=target_pdb_path)
    print(f"SCORE: {score} --   MIRROR SCORE: {mirror_score}")

    create_video(
        target_coords=seq.coords,
        recording=spine_recording + recording,
        seq_str=seq.seq_str,
        speed=VIDEO_SPEED,
        save_path="videos/Sequence.mp4",
        interval=33,
    )
    print(f"SCORE: {score} --   MIRROR SCORE: {mirror_score}")

    plot_coords_list(
        seq_str=seq.seq_str,
        coords = adjusted_coords,
        reference_coords=seq.coords,
    )


if __name__ == "__main__":
    # ================ Test 1 ==============
    # print("Testing one sequence")
    # analyse_one()
    # =============== Test 2 ==============

    # print("Testing multiple sequence lengths scores")
    analyse_scores(20)

    # =============== Test 3 ==============
    # print("Loading Sequence Dataset")
    # seq_dataset = SequenceDataset(n_sequences=N_SEQUENCES, subset_len=SEQUENCE_SIZE)
    # print(len(seq_dataset.subset_sequences))
    # # Initialize a real sequence (Distance Matrix Assigned)
    # # seq_dataset.get_stats()
    # print("Loading Random Sequence")
    # # seq = seq_dataset.get_random_subset()
    # seq = seq_dataset.subset_sequences[0]
    # print(f"Sequence Length: {len(seq.seq_str)}")
    # # seq._coords_to_noise()
    # # seq.adjust_coords(n_iter=100, use_neighbors=USE_NEIGHBORS)
    # seq._primary_test()

    # Sequence.compute_similarity_us_align()
