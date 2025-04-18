from torch import Tensor
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.pyplot as plt


EPS = 1e-8

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
    max_delta: float = 0.01,
) -> Tensor:
    new_distance_matrix = coord_to_distance_matrix(coord_matrix)
    dist_diff = new_distance_matrix - target_distance_matrix
    d_coords = coord_matrix.unsqueeze(0) - coord_matrix.unsqueeze(1)

    u_vecs = d_coords / (new_distance_matrix.unsqueeze(2) + EPS)
    deltas = (u_vecs * dist_diff.unsqueeze(2)).sum(dim=1)
    mags = torch.norm(deltas, dim=1, keepdim=True)
    scale = max_delta / (mags + EPS)
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

   
def align(input_coords:list[Vector], target_coords:list[Vector]):
    """
    1. Use the first 3 coordinates to create a plane for self.coords and target_coords.
    2. Create a quaternion from the planes using SciPy to align the plane normal of self.coords
    to that of target_coords.
    3. Rotate self.coords by the quaternion.
    4. Translate all points so that self.coords[0] aligns with target_coords[0].
    5. Align the unit vector from self.coords[0] to self.coords[1] with the unit vector from
    target_coords[0] to target_coords[1] (apply an additional twist rotation about the plane normal).
    6. Return the new coordinates as a list of Vector objects.
    """

    if len(input_coords) < 3 or len(target_coords) < 3:
        raise ValueError("Need at least 3 points in each set.")

    # Build numpy arrays
    def to_np(v):
        return np.array([v.x, v.y, v.z], dtype=float)

    p0, p1, p2 = map(to_np, input_coords[:3])
    q0, q1, q2 = map(to_np, target_coords[:3])

    # Source normal
    v1, v2 = p1 - p0, p2 - p0
    n_source = np.cross(v1, v2)
    norm_s = np.linalg.norm(n_source)

    # Target normal
    w1, w2 = q1 - q0, q2 - q0
    n_target = np.cross(w1, w2)
    norm_t = np.linalg.norm(n_target)

    # Normalize if safe, otherwise mark as degenerate
    if norm_s > EPS:
        n_source_norm = n_source / norm_s
    else:
        n_source_norm = None

    if norm_t > EPS:
        n_target_norm = n_target / norm_t
    else:
        n_target_norm = None

    # STEP 2: Align normals with try/except
    if n_source_norm is not None and n_target_norm is not None:
        try:
            # Note: align_vectors takes (target, source)
            rot_obj, _ = R.align_vectors([n_target_norm], [n_source_norm])
        except ValueError:
            # Degenerate quaternion; fall back to identity
            rot_obj = R.identity()
    else:
        # One of the normals was degenerate
        rot_obj = R.identity()

    # STEP 3: Rotate all input points
    pts = np.vstack([to_np(v) for v in input_coords])
    rotated = rot_obj.apply(pts)

    # STEP 4: Translate so first points coincide
    translation = q0 - rotated[0]
    aligned = rotated + translation

    # STEP 5: Twist alignment of first‐to‐second vector
    # Source direction
    ds = aligned[1] - aligned[0]
    dn_s = ds / (np.linalg.norm(ds) + EPS)
    # Target direction
    dt = q1 - q0
    dn_t = dt / (np.linalg.norm(dt) + EPS)

    # Compute signed angle
    dot = np.clip(np.dot(dn_s, dn_t), -1.0, 1.0)
    angle = np.arccos(dot)
    # Use the (possibly degenerate) target normal to sign the twist
    axis = n_target_norm if n_target_norm is not None else np.array([0, 0, 1])
    sign = np.sign(np.dot(np.cross(dn_s, dn_t), axis))
    twist = R.from_rotvec(axis * (angle * sign))

    final_pts = np.array([q0 + twist.apply(pt - q0) for pt in aligned])

    # Convert back to Vectors
    return [Vector(x, y, z) for x, y, z in final_pts]

def plot_coords_list(coords_list: list[list[Vector]], set_names: list[str] = None):
    """
    Plot multiple sequences by marking each point and connecting each coordinate
    i to i+1 with a line. Each vector set in coords_list is plotted as a separate
    line using a distinct color. The line is rendered with added transparency and
    thickness, and the points are larger.

    Parameters:
    coords_list (list[list[Vector]]): A list containing one or more lists of Vector objects.
                                        If None, uses self.coords as a single vector set.
    set_names (list[str]): Optional list of labels for each coordinate set. The length of
                            set_names must match the number of coordinate sets.
    """

    coords_list = [align(coords, coords_list[0]) for coords in coords_list]

    # If no set names are provided, use default names.
    if set_names is None:
        set_names = [f"Set {i+1}" for i in range(len(coords_list))]
    elif len(set_names) != len(coords_list):
        raise ValueError(
            "Length of set_names must equal the number of coordinate sets in coords_list"
        )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Use a colormap to assign a distinct color to each coordinate set.
    cmap = matplotlib.colormaps.get_cmap("tab10")

    # Iterate over each vector set and plot the points and connecting line.
    for i in range(len(coords_list)):
        coords = coords_list[i]
        x = [coord.x for coord in coords]
        y = [coord.y for coord in coords]
        z = [coord.z for coord in coords]

        color = cmap(i)  # Get a distinct color for the current set

        # Scatter plot the points with increased size.
        ax.scatter(x, y, z, s=100, color=color, label=f"{set_names[i]} Points")

        # Connect the points with a line that is thicker and partially transparent.
        ax.plot(
            x,
            y,
            z,
            color=color,
            alpha=0.7,
            linewidth=2,
            label=f"{set_names[i]} Path",
        )

    # Label the axes.
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Add a title and legend for clarity.
    ax.set_title("3D Vector Plot")
    ax.legend()

    plt.show()