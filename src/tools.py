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


def apply_heat(distance_matrix: Tensor, temperature: float) -> Tensor:
    """
    Apply Gaussian noise to the distance matrix.
    """
    noise = torch.normal(0, temperature, size=distance_matrix.shape)
    # Add noise to the distance matrix
    distance_matrix += noise
    return distance_matrix


def drop(tensor: Tensor, drop_rate: float) -> Tensor:
    """
    Randomly drop elements from a tensor with a given probability.
    """
    mask = torch.rand(tensor.shape) > drop_rate
    tensor = tensor * mask
    return tensor


def adjust_coords(
    n_iter: int,
    coords: list[Vector],
    target_matrix: Tensor,
    temperature: float,
    eps: float,
    delta_drop_rate: float,
    max_delta: float,
) -> list[Vector]:
    """
    Make coords match the distance matrix via simulation.
    1. Calculate distance matrix from current coordinates.
    2. Calculate difference between the current and target distance matrices.
    3. Create list of adjustments for each coordinate.
    4. Calculate unit vectors from coord i to coord j.
    Only contributions from pairs with distances <= MAX_DISTANCE are considered.
    """
    coord_matrix = coords_list_to_matrix(coords)  # Changes every iteration
    recording = []

    for curr in range(n_iter):
        print(f"Iteration {curr+1}/{n_iter}")

        # 1. Initiate Target Structure Via Distance Matrix
        target_distance_matrix = target_matrix.clone()
        # 2. Apply Heat to the Structure.
        target_distance_matrix = apply_heat(target_distance_matrix, temperature)
        # 3. Compute Deltas Based on Target Distance Matrix
        deltas = compute_delta_matrix(
            coord_matrix=coord_matrix,
            target_distance_matrix=target_distance_matrix,
            eps=eps,
            max_delta=max_delta,
        )
        # 3.1. Drop some deltas to simulate imperfect information
        deltas = drop(deltas, drop_rate=delta_drop_rate)
        # 4. Add the deltas to the coordinates.
        coord_matrix = coord_matrix + deltas
        # 5. Record the current state.
        recording.append(coord_matrix.clone())

    # Update self.coords from the coord_matrix.
    for i in range(len(coords)):
        coords[i] = Vector(coord_matrix[i][0], coord_matrix[i][1], coord_matrix[i][2])

    return coords, recording

