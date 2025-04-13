import os
from typing import Literal
from torch import Tensor
from torch import nn
import pandas as pd
import random

# set here to cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ====== CONSTANTS ======
CLUSTER_SIZE = 4
LABELS_PATH = "data/train_labels.csv"
SEQUENCES_PATH = "data/train_sequences.csv"

# ====== TYPES ======
class Vector:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        
    def copy(self):
        return Vector(self.x, self.y, self.z),

# ====== SEQUENCE ======
class Sequence:
    """
    Contains a type encodeding of a sequence.
    """

    def __init__(self, str: str, coords: list[Vector] = None):
        self.encoding = self.encode_str(str)
        self.coords = coords if coords else self.generate_random_coords(len(str))
        self.distance_matrix = self.compute_distance_matrix(coords) if coords else None

    def __repr__(self):
        msg = "\n"
        msg += " ========= SEQUENCE ======== \n"
        msg += "\n"
        msg += "ENCODING: \n"
        msg += f"{self.encoding} \n"
        msg += "---------------------------- \n"
        msg += "\n"
        msg += "DISTANCE MATRIX: \n"
        msg += f"{self.distance_matrix} \n"
        msg += "---------------------------- \n"
        return msg

    @staticmethod
    def encode_str(str: str):
        """
        Convert a string to hot-encoded tensor.
        "A" -> [1, 0, 0, 0]
        "C" -> [0, 1, 0, 0]
        "G" -> [0, 0, 1, 0]
        "T" -> [0, 0, 0, 1]
        "N" -> [0, 0, 0, 0]        Tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        """
        encoding = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
        }
        tensor = []
        for char in str:
            if char not in encoding:
                tensor.append([0, 0, 0, 0])
                continue
            tensor.append(encoding[char])
        assert len(tensor) == len(
            str
        ), "Length of tensor does not match length of string"
        assert len(tensor[0]) == 4, "Length of tensor does not match length of encoding"
        return Tensor(tensor)

    @staticmethod
    def compute_distance_matrix(coord_list: list[Vector]) -> Tensor:
        """
        Compute the distance matrix for a list of coordinates.
        :param coords: List of coordinates
        :return: Distance matrix
        """
        distance_matrix = []
        for i in range(len(coord_list)):
            row = []
            for j in range(len(coord_list)):
                if i == j:
                    row.append(0)
                else:
                    row.append(Sequence.distance(coord_list[i], coord_list[j]))
            distance_matrix.append(row)
        return Tensor(distance_matrix)

    @staticmethod
    def distance(coord1: Vector, coord2: Vector) -> float:
        """
        Compute the distance between two coordinates.
        :param coord1: Coordinate 1
        :param coord2: Coordinate 2
        :return: Distance
        """
        return (
            (coord1.x - coord2.x) ** 2
            + (coord1.y - coord2.y) ** 2
            + (coord1.z - coord2.z) ** 2
        ) ** 0.5
    @staticmethod
    def adjust_coords(self):
        """ 
        1. Calculate distance matrix from current coordinates
        2. Calc difference between distance matrix and target distance matrix
        3. Create list of dx, dy, dz for each coordinate 
        4. Calc all unit vectors from coord i to coord j
        
        """
        new_distance_matrix = self.compute_distance_matrix(self.coords)
        diff = new_distance_matrix - self.distance_matrix
        # adjust coordinates based on diff
        k = 0.1
        deltas = []
        for i in range(len(self.coords)):
            for j in range(len(self.coords)):
                if i == j:
                    continue
                dx = self.coords[j].x - self.coords[i].x
                dy = self.coords[j].y - self.coords[i].y
                dz = self.coords[j].z - self.coords[i].z
                dist = Sequence.distance(self.coords[i], self.coords[j])
                unit_vector = Vector(dx / dist, dy / dist, dz / dist)
                deltas.append(
                    Vector(
                        unit_vector.x * diff[i][j] * k,
                        unit_vector.y * diff[i][j] * k,
                        unit_vector.z * diff[i][j] * k,
                    )
                )
        for i in range(len(self.coords)):
            self.coords[i].x += deltas[i].x
            self.coords[i].y += deltas[i].y
            self.coords[i].z += deltas[i].z
    @staticmethod
    def generate_random_coords(n: int = 4) -> list[Vector]:
        """
        Generate random coordinates.
        first anywhere
        then following go 5.5A in a 3d direction
        """
        r = 5.5
        coords = []
        curr_pos = Vector(
            random.uniform(-r, r),
            random.uniform(-r, r),
            random.uniform(-r, r),
        )
        while len(coords) < n:
            magnitude = random.uniform(0, r)
            theta = random.uniform(0, 2 * 3.14)
            phi = random.uniform(0, 2 * 3.14)
            x = curr_pos.x + magnitude * (r * theta)
            y = curr_pos.y + magnitude * (r * phi)
            z = curr_pos.z + magnitude * (r * theta * phi)
            coords.append(Vector(x, y, z))
            curr_pos = Vector(x, y, z)
        return coords

# ====== DATA PPEPERATION ======
class SequenceDataset:
    """
    Inputs is the Sequence Encoding
    Output is the distance matrix
    """

    real_sequences: list[Sequence]

    def __init__(self, n_sequences: int = 10000):
        self.real_sequences = self.get_real_sequences(n_sequences)

    def get_real_sequences(self, target_n_sequences: int):
        label_df = pd.read_csv(LABELS_PATH)
        # using a windowed approach
        sequences = []

        max_length = len(label_df)
        curr_index = 0

        while (
            len(sequences) < target_n_sequences
            or curr_index + CLUSTER_SIZE > max_length
        ):
            i = curr_index

            resid = label_df.iloc[i]["resid"]
            end_resid = label_df.iloc[i + CLUSTER_SIZE]["resid"]
            if resid > end_resid:
                curr_index += CLUSTER_SIZE
                continue
            labels_window = label_df.iloc[i : i + CLUSTER_SIZE]
            seq_window = labels_window["resname"].tolist()
            seq_str = "".join(seq_window)
            x_1 = labels_window["x_1"].tolist()
            y_1 = labels_window["y_1"].tolist()
            z_1 = labels_window["z_1"].tolist()
            coords = []
            for x, y, z in zip(x_1, y_1, z_1):
                coords.append(Vector(x, y, z))
            seq = Sequence(seq_str, coords)
            sequences.append(seq)
            curr_index += 1
        self.real_sequences = sequences
        [print(seq) for seq in sequences[:5]]
        return sequences


# ======== MODELS ==========
class DistanceMatrixModel(nn.Module):
    """
    Takes in a sequence and outputs a distance matrix.
    """

    def __init__():
        super().__init__()


if __name__ == "__main__":
    # Test the Sequence class
    seq = Sequence("ACGTAA8")
    print(seq)

    seq_dataset = SequenceDataset()
    print(len(seq_dataset.real_sequences))
