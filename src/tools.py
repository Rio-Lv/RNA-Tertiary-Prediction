from torch import Tensor
import torch


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
        return (Vector(self.x, self.y, self.z),)

    def add(self, vector: "Vector"):
        self.x += vector.x
        self.y += vector.y
        self.z += vector.z

    def __repr__(self):
        return f"Vector({self.x}, {self.y}, {self.z}) \n"

# ====== FUNCTIONS ======
def encode_str(seq_str: str):
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
    for char in seq_str:
        if char not in encoding:
            tensor.append([0, 0, 0, 0])
            continue
        tensor.append(encoding[char])
    assert len(tensor) == len(
        seq_str
    ), "Length of tensor does not match length of string"
    assert len(tensor[0]) == 4, "Length of tensor does not match length of encoding"
    return Tensor(tensor)


def coords_list_to_matrix(coords: list[Vector]) -> Tensor:
    """
    Create a tensor from a list of coordinates.
    :param coords: List of coordinates
    :return: Tensor of coordinates
    """
    coord_matrix = []
    for coord in coords:
        coord_matrix.append([coord.x, coord.y, coord.z])
    return Tensor(coord_matrix)


def coord_matrix_to_list(coord_matrix: Tensor) -> list[Vector]:
    """
    Create a list of coordinates from a tensor.
    :param coord_matrix: Tensor of coordinates
    :return: List of coordinates
    """
    coords = []
    for coord in coord_matrix:
        coords.append(Vector(coord[0], coord[1], coord[2]))
    return coords


def coord_to_distance_matrix(coord_matrix: Tensor) -> Tensor:
    """
    Compute the distance matrix from a coordinate matrix.
    :param coord_matrix: Tensor of shape [N, 3] containing the coordinates.
    :return: Tensor of shape [N, N] representing the distance matrix.
    """
    # Compute pairwise distances using broadcasting
    dist = torch.norm(coord_matrix.unsqueeze(0) - coord_matrix.unsqueeze(1), dim=2)

    return dist


def compute_delta_matrix(
    coord_matrix: Tensor,
    target_distance_matrix: Tensor,
    eps: float = 1e-8,
    max_delta: float = 0.01,
) -> Tensor:
    new_distance_matrix = coord_to_distance_matrix(coord_matrix)
    dist_diff = new_distance_matrix - target_distance_matrix
    d_coords = coord_matrix.unsqueeze(0) - coord_matrix.unsqueeze(1)

    u_vecs = d_coords / (new_distance_matrix.unsqueeze(2) + eps)
    deltas = (u_vecs * dist_diff.unsqueeze(2)).sum(dim=1)
    mags = torch.norm(deltas, dim=1, keepdim=True)
    scale = max_delta / (mags + eps)
    deltas = deltas * scale

    return deltas
