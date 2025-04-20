import os
import time
import subprocess
import re
from typing import Optional
from torch import Tensor
import torch 
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple
import time
import matplotlib.animation as animation

EPS = 1e-8
VIDEO_PADDING = 0.3
MIN_BOUNCE_DISTANCE = 5  # Minimum distance between atoms after bounce

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
        return Vector(self.x, self.y, self.z)

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


def plot_distance_heatmap(
    distance_matrix: torch.Tensor, title: str = "Distance matrix"
) -> None:
    """Plot a square heat-map for a (LxL) distance matrix."""
    plt.figure()
    plt.imshow(distance_matrix.cpu(), aspect="equal")
    plt.colorbar(label="Distance")
    plt.title(title)
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.tight_layout()
    plt.show()


def compute_delta_matrix(
    coord_matrix: Tensor,
    target_distance_matrix: Tensor,
    max_delta: float = 0.1,
) -> Tensor:
    # mask out values where target distance is less than 1
    mask = target_distance_matrix > 1

    curr_distance_matrix = coord_to_distance_matrix(coord_matrix)
    dist_diff = curr_distance_matrix - target_distance_matrix
    dist_diff = dist_diff * mask
    d_coords = coord_matrix.unsqueeze(0) - coord_matrix.unsqueeze(1)

    u_vecs = d_coords / (curr_distance_matrix.unsqueeze(2) + EPS)
    deltas = (u_vecs * dist_diff.unsqueeze(2)).sum(dim=1)
    mags = torch.norm(deltas, dim=1, keepdim=True)
    scale = max_delta / (mags + EPS)
    deltas = deltas * scale

    return deltas


def atomic_bounce(
    coord_matrix: Tensor, max_delta: float, min_distance: float = 3.5
) -> Tensor:
    """
    Push apart atoms that are closer than `min_distance`.

    Parameters
    ----------
    coord_matrix : Tensor  [N, 3]
        Cartesian coordinates (in Å) of N atoms.
    min_distance : float
        Minimum allowed inter‑atomic distance.

    Returns
    -------
    Tensor  [N, 3]
        New coordinates after a single bounce step.
    """
    # Pair‑wise displacement vectors  [N, N, 3]
    diffs = coord_matrix[:, None, :] - coord_matrix[None, :, :]

    # Euclidean distances  [N, N]
    dists = torch.linalg.norm(diffs + 1e-12, dim=-1)  # small ε avoids division by zero

    # Mask of offending pairs (exclude i==j on the diagonal)
    mask = (dists < min_distance) & (dists > 0)

    if not mask.any():  # nothing to do
        return coord_matrix

    # Unit vectors pointing from j → i   [N, N, 3]
    directions = diffs / dists.unsqueeze(-1)

    # How far each member of the pair needs to move (scalar)  [N, N]
    delta = (min_distance - dists) / 2.0  # (Å)

    # Zero‑out pairs we are not fixing
    delta = delta * mask

    # Vector displacement for *each* ordered pair  [N, N, 3]
    pair_shifts = directions * delta.unsqueeze(-1)

    # Net shift for each atom = sum of contributions from all partners  [N, 3]
    atom_shifts = pair_shifts.sum(dim=1)
    
    # Scale the shifts to be within max_delta
    atom_shifts_mags = torch.norm(atom_shifts, dim=1, keepdim=True)
    scale = max_delta / (atom_shifts_mags + EPS)
    atom_shifts = atom_shifts * scale
    atom_shifts = torch.clamp(atom_shifts, -max_delta, max_delta)

    # Apply the shifts
    return coord_matrix + atom_shifts 


def apply_heat(distance_matrix: Tensor, temperature: float) -> Tensor:
    """
    Apply Gaussian noise to the distance matrix.
    """
    noise = torch.normal(0, temperature, size=distance_matrix.shape)
    # Add noise to the distance matrix
    distance_matrix += noise
    return distance_matrix


def drop_random(active_distances: Tensor, keep_rate: float) -> Tensor:
    """
    To be used on deltas which is of shape [N, N, 3].
    Randomly drop elements from a tensor with a given probability.
    """
    mask = torch.rand(active_distances.shape) < keep_rate
    active_distances = active_distances * mask
    return active_distances


def create_index_drop_mask(distance_matrix: Tensor, max_index_diff: int) -> Tensor:
    """
    Create a mask to drop elements from the distance matrix based on index difference.
    """
    mask = torch.ones_like(distance_matrix, dtype=torch.bool)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if abs(i - j) > max_index_diff:
                mask[i, j] = False
    return mask


def sub_next_coord(active_coord_matrix: Tensor) -> Tensor:
    """
    Replace index 0 of this matrix with intelligent placement.
    This is a placeholder function.
    for now..
    1. calculate vector coord index 2 - coord index 1
    2. subtract vector from index 1 to get index 0 coord
    - 0 to 1 to 2 should be a straight line
    """
    # r = 3.5 # distance between atoms
    r = 0.1
    # Calculate the vector from coord 1 to coord 2
    vector = active_coord_matrix[-2] - active_coord_matrix[-1]
    dist = torch.norm(vector)
    uv = vector / (dist + EPS)  # unit vector
    # Subtract the vector from coord 1 to get coord 0
    new_coord = active_coord_matrix[0] - (uv * r)
    # Replace coord 0 with the new coord
    active_coord_matrix[0] = new_coord

    return active_coord_matrix


def adjust_coords(
    n_iter: int,
    input_coords: list[Vector],
    target_matrix: Tensor,
    temperature: float,
    active_keep_rate: float,
    max_delta: float,
    iterations_per_residue: int = None,
    # adjust_last: bool = False,
) -> Tuple[list[Vector], list[Tensor]]:
    """
    Make input_coords match the distance matrix via simulation.
    1. Calculate distance matrix from current coordinates.
    2. Calculate difference between the current and target distance matrices.
    3. Create list of adjustments for each coordinate.
    4. Calculate unit vectors from coord i to coord j.
    Only contributions from pairs with distances <= MAX_DISTANCE are considered.
    """
    coord_matrix = coords_list_to_matrix(input_coords)  # Changes every iteration
    recording = []
    length, _ = coord_matrix.shape

    max_index = length
    for curr in range(n_iter):

        if iterations_per_residue:
            max_index = min(curr // iterations_per_residue + 4, length)

        active_coord_matrix = coord_matrix[:max_index].clone()  # rows only
        
        if len(active_coord_matrix) < length:
            active_coord_matrix = sub_next_coord(active_coord_matrix)

        if curr % 500 == 0 and curr >= 500:
            print(f"Iteration {curr }/{n_iter}")

        # 1. Initiate Target Structure Via Distance Matrix
        active_distance_matrix = target_matrix.clone()[:max_index, :max_index]
        active_distance_matrix = active_distance_matrix
        # 2. Apply Heat to the Structure.
        active_distance_matrix = apply_heat(active_distance_matrix, temperature)
        # 2.1. Drop some deltas to simulate imperfect information
        active_distance_matrix = drop_random(
            active_distance_matrix, keep_rate=active_keep_rate
        )

        # 3. Compute Deltas Based on Target Distance Matrix
        deltas = compute_delta_matrix(
            coord_matrix=active_coord_matrix,
            target_distance_matrix=active_distance_matrix,
            max_delta=max_delta,
        )
        # 4. Add the deltas to the coordinates.
        active_coord_matrix[:max_index, :max_index] += deltas
        
        # 5. Apply atomic bounce to the coordinates.
        active_coord_matrix = atomic_bounce(
            active_coord_matrix, max_delta, min_distance=MIN_BOUNCE_DISTANCE
        )
        # if adjust_last:
        #     coord_matrix[max_index, max_index] = active_coord_matrix[-1, -1]
        # else:
        coord_matrix[:max_index, :max_index] = active_coord_matrix
        # 6. Record the current state.
        recording.append(active_coord_matrix.clone())

    # Update self.coords from the coord_matrix.
    for i in range(len(input_coords)):
        input_coords[i] = Vector(
            coord_matrix[i][0], coord_matrix[i][1], coord_matrix[i][2]
        )

    return input_coords, recording


def align(input_coords: list[Vector], target_coords: list[Vector]):
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

def mirror(coords: list[Vector], axis: str = "x") -> list[Vector]:
    """
    Mirror the coordinates across a specified axis.
    :param coords: List of Vector objects to be mirrored.
    :param axis: Axis to mirror across ('x', 'y', or 'z').
    :return: List of mirrored Vector objects.
    """
    mirrored_coords = []
    for coord in coords:
        if axis == "x":
            mirrored_coords.append(Vector(-coord.x, coord.y, coord.z))
        elif axis == "y":
            mirrored_coords.append(Vector(coord.x, -coord.y, coord.z))
        elif axis == "z":
            mirrored_coords.append(Vector(coord.x, coord.y, -coord.z))
        else:
            raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")
    return mirrored_coords

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
    print(coords_list)
    if len(coords_list) > 1:
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

def create_video(
    target_coords: list[Vector],
    recording: list[Tensor],
    speed: int,
    save_path: str,
    interval: int = 33,
):
    start_time = time.time()
    recording_coords: list[list[Vector]] = []
    for i in range(len(recording)):
        if i % speed != 0:
            continue
        frame_tensor = recording[i]
        frame_coords = [
            Vector(coord[0].item(), coord[1].item(), coord[2].item())
            for coord in frame_tensor
        ]
        recording_coords.append(frame_coords)

    # Downsample the recording so that only every Nth frame is rendered.
    recording_coords = recording_coords
    num_frames = len(recording_coords)
    # Compute bounding box limits based on the original coordinates with 50% padding.
    
    x_orig_vals = [coord.x for coord in target_coords]
    y_orig_vals = [coord.y for coord in target_coords]
    z_orig_vals = [coord.z for coord in target_coords]

    x_min, x_max = min(x_orig_vals), max(x_orig_vals)
    y_min, y_max = min(y_orig_vals), max(y_orig_vals)
    z_min, z_max = min(z_orig_vals), max(z_orig_vals)

    # Ensure nonzero ranges.
    x_range = x_max - x_min if (x_max - x_min) != 0 else 1.0
    y_range = y_max - y_min if (y_max - y_min) != 0 else 1.0
    z_range = z_max - z_min if (z_max - z_min) != 0 else 1.0

    x_pad = VIDEO_PADDING * x_range
    y_pad = VIDEO_PADDING * y_range
    z_pad = VIDEO_PADDING * z_range

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
        x_orig = [coord.x for coord in target_coords]
        y_orig = [coord.y for coord in target_coords]
        z_orig = [coord.z for coord in target_coords]
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
        current_coords = align(recording_coords[frame], target_coords)
        x_adj = [coord.x for coord in current_coords]
        y_adj = [coord.y for coord in current_coords]
        z_adj = [coord.z for coord in current_coords]

        ax.clear()
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)

        # Plot the static original coordinates.
        x_orig = [coord.x for coord in target_coords]
        y_orig = [coord.y for coord in target_coords]
        z_orig = [coord.z for coord in target_coords]
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
    ani.save(save_path, writer=writer)
    plt.close(fig)

    elapsed_time = time.time() - start_time
    print(f"Video saved to {save_path} in {elapsed_time:.2f} seconds")


# ============= Compute Similarity =============
def compute_similarity(path_1: str, path_2: str, timeout: float = 3.0) -> Optional[float]:
    """
    Run USalign on two PDB files, capture its output, and return the TM-score
    normalized by the length of Structure_1.

    :param path_1: Path to Structure_1 PDB.
    :param path_2: Path to Structure_2 PDB.
    :param timeout: Max seconds to wait for files to exist.
    :return: TM-score (float) or None if not found / on error.
    """
    # wait for files up to `timeout`
    start = time.time()
    while not (os.path.exists(path_1) and os.path.exists(path_2)):
        if time.time() - start > timeout:
            print(f"Error: files not found within {timeout}s: {path_1}, {path_2}")
            return None
        time.sleep(0.1)

    # run USalign and capture output
    try:
        result = subprocess.run(
            ["../USalign/USalign", path_1, path_2],
            capture_output=True,
            text=True
        )
    except Exception as e:
        print(f"Error running USalign: {e}")
        return None

    stdout = result.stdout

    # regex to find the TM-score normalized by Structure_1
    m = re.search(
        r"TM-score=\s*([0-9]+(?:\.[0-9]+)?)\s*\(normalized by length of Structure_1",
        stdout
    )
    if not m:
        print("TM-score (normalized by Structure_1) not found in USalign output.")
        return None
    return float(m.group(1))