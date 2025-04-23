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
from typing import Tuple, List
import time
import matplotlib.animation as animation


if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


EPS = 1e-8
VIDEO_PADDING = 0.25
MIN_BOUNCE_DISTANCE = 3.5  # Minimum distance between atoms after bounce
A_COLOR = "red"
C_COLOR = "blue"
G_COLOR = "green"
U_COLOR = "orange"



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

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_array(a: np.ndarray) -> "Vector":
        return Vector(float(a[0]), float(a[1]), float(a[2]))

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
    return Tensor(tensor).to(device)


def coords_list_to_matrix(coords: list[Vector]) -> Tensor:
    """
    Create a tensor from a list of coordinates.
    :param coords: List of coordinates
    :return: Tensor of coordinates
    """
    coord_matrix = []
    for coord in coords:
        coord_matrix.append([coord.x, coord.y, coord.z])
    return Tensor(coord_matrix).to(device)


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

    return dist.to(device)


def plot_distance_heatmap(
    distance_matrix: torch.Tensor, title: str = "Distance matrix"
) -> None:
    distance_matrix = distance_matrix.cpu()
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

    return deltas.to(device)


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
    noise = torch.normal(0, temperature, size=distance_matrix.shape, device=device)
    # Add noise to the distance matrix
    distance_matrix += noise
    return distance_matrix


def drop_random(active_distances: Tensor, keep_rate: float) -> Tensor:
    """
    To be used on deltas which is of shape [N, N, 3].
    Randomly drop elements from a tensor with a given probability.
    """
    mask = torch.rand(active_distances.shape, device=device) < keep_rate
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
    seq_str: str,
    input_coords: list[Vector],
    target_matrix: Tensor,
    temperature: float,
    active_keep_rate: float,
    max_delta: float,
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

        active_coord_matrix = coord_matrix[:max_index].clone()  # rows only

        if curr % 500 == 0 and curr >= 500:
            print(f"Iteration {curr }/{n_iter}")

        # 1. Initiate Target Structure Via Distance Matrix
        active_distance_matrix = target_matrix.clone()[:max_index, :max_index]
        active_distance_matrix = active_distance_matrix
        # 2. Apply Heat to the Structure.
        active_distance_matrix = apply_heat(active_distance_matrix, temperature)
        
        # 2.2. Drop some deltas to simulate imperfect information
        active_distance_matrix = drop_random(
            active_distance_matrix, keep_rate=active_keep_rate
        )

        # 3. Compute Deltas Based on Target Distance Matrix
        deltas = compute_delta_matrix(
            coord_matrix=active_coord_matrix,
            target_distance_matrix=active_distance_matrix,
            max_delta=max_delta,
            # stability_matrix = stability_matrix,
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

def plot_coords_list(
    seq_str: str,
    coords: list[Vector],
    reference_coords: list[Vector] = None,
):
    """
    Plot a 3D sequence:
      • reference_coords in faint gray (alpha=0.5) if provided
      • coords path in black
      • points colored by seq_str bases (A/C/G/U)
    """
    # If you have a reference, align your coords to it
    if reference_coords is not None:
        coords = align(coords, reference_coords)

    N = len(seq_str)
    if len(coords) != N:
        raise ValueError(f"`coords` length ({len(coords)}) != seq_str length ({N})")
    if reference_coords is not None and len(reference_coords) != N:
        raise ValueError("`reference_coords` must match length of `seq_str`")

    # Helper to extract Python floats from Vector components
    def to_float(x):
        # if it's a tensor, bring to CPU then .item()
        if hasattr(x, "cpu"):
            return x.detach().cpu().item()
        return float(x)

    # Convert all coords to lists of floats up front
    coords_f = [Vector(to_float(v.x), to_float(v.y), to_float(v.z)) for v in coords]
    ref_f = None
    if reference_coords is not None:
        ref_f = [Vector(to_float(v.x), to_float(v.y), to_float(v.z)) for v in reference_coords]

    # Base→color mapping
    base_color_map = {"A": A_COLOR, "C": C_COLOR, "G": G_COLOR, "U": U_COLOR}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 1) reference path in gray
    if ref_f is not None:
        xr = [v.x for v in ref_f]
        yr = [v.y for v in ref_f]
        zr = [v.z for v in ref_f]
        ax.scatter(xr, yr, zr, color="gray", alpha=0.5, s=100, label="Reference Points")
        ax.plot(xr, yr, zr, color="gray", alpha=0.5, linewidth=2, label="Reference Path")

    # 2) main path in black
    x = [v.x for v in coords_f]
    y = [v.y for v in coords_f]
    z = [v.z for v in coords_f]
    ax.plot(x, y, z, color="black", alpha=0.7, linewidth=2, label="Coords Path")

    # 3) scatter by base
    base_groups: dict[str, list[Vector]] = {b: [] for b in base_color_map}
    for base, v in zip(seq_str, coords_f):
        if base in base_groups:
            base_groups[base].append(v)

    for base, group in base_groups.items():
        if not group:
            continue
        xb = [v.x for v in group]
        yb = [v.y for v in group]
        zb = [v.z for v in group]
        ax.scatter(xb, yb, zb,
                   color=base_color_map[base],
                   s=100,
                   label=f"{base}")

    # Labels & legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Vector Plot")
    plt.subplots_adjust(right=0.75)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), borderaxespad=0.0)
    plt.show()
    
def create_video(
    target_coords: list[Vector],
    recording: list[Tensor],
    seq_str: str,
    speed: int,
    save_path: str,
    interval: int = 33,
):
    """
    Animate the adjustment process from `recording` towards `target_coords`.

    - recording: list of frames, each either a torch.Tensor (N×3) or a numpy array
    - target_coords: list of Vector(x,y,z), possibly holding .x/.y/.z as tensors
    """

    assert len(seq_str) == len(target_coords), "seq_str must match target length"
    start_time = time.time()

    # ─── 1) Coerce ALL recorded frames to numpy arrays ────────────────
    rec_np: list[np.ndarray] = []
    for frame in recording:
        if isinstance(frame, np.ndarray):
            rec_np.append(frame)
        else:
            # torch.Tensor -> CPU numpy
            rec_np.append(frame.detach().cpu().numpy())

    # ─── 2) Coerce target_coords to Python floats ────────────────────
    def to_vector_f(v: Vector) -> Vector:
        # if components are tensors, extract .item(), else cast float()
        x = v.x.cpu().item() if hasattr(v.x, "cpu") else float(v.x)
        y = v.y.cpu().item() if hasattr(v.y, "cpu") else float(v.y)
        z = v.z.cpu().item() if hasattr(v.z, "cpu") else float(v.z)
        return Vector(x, y, z)

    tgt_f: list[Vector] = [to_vector_f(v) for v in target_coords]

    # ─── 3) Decimate and build per-frame Vector lists ───────────────
    recording_coords: list[list[Vector]] = []
    for idx, arr in enumerate(rec_np):
        if idx % speed != 0:
            continue
        # arr is now a pure NumPy array of shape (N,3)
        recording_coords.append([Vector(x, y, z) for x, y, z in arr])

    num_frames = len(recording_coords)

    # ─── 4) Chirality correction on the last frame ────────────────
    _, flip = correct_chirality(recording_coords[-1])
    if flip:
        recording_coords = [mirror(frame) for frame in recording_coords]

    # ─── 5) Compute bounds from tgt_f as plain floats ───────────────
    xs = [v.x for v in tgt_f]
    ys = [v.y for v in tgt_f]
    zs = [v.z for v in tgt_f]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    x_pad = VIDEO_PADDING * (x_max - x_min or 1.0)
    y_pad = VIDEO_PADDING * (y_max - y_min or 1.0)
    z_pad = VIDEO_PADDING * (z_max - z_min or 1.0)

    # All limits are now pure floats
    x_lim = (x_min - x_pad, x_max + x_pad)
    y_lim = (y_min - y_pad, y_max + y_pad)
    z_lim = (z_min - z_pad, z_max + z_pad)

    # ─── 6) Color map & figure setup ───────────────────────────────
    base_colors = {"A": A_COLOR, "C": C_COLOR, "G": G_COLOR, "U": U_COLOR}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def init():
        ax.clear()
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_zlim(*z_lim)
        # faint target
        ax.scatter(xs, ys, zs, color="black", s=100, alpha=0.05)
        ax.plot(xs, ys, zs, color="black", alpha=0.05, linewidth=2)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("Adjustment Process")
        return []

    def update(frame_idx: int):
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx+1}/{num_frames}")

        current = align(recording_coords[frame_idx], tgt_f)

        # group by base
        groups = {b: [] for b in base_colors}
        for base, vec in zip(seq_str, current):
            groups[base].append(vec)

        ax.clear()
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_zlim(*z_lim)

        # re‐plot faint target
        ax.scatter(xs, ys, zs, color="black", s=100, alpha=0.05)
        ax.plot(xs, ys, zs, color="black", alpha=0.05, linewidth=2)

        # plot current by base
        for base, color in base_colors.items():
            pts = groups[base]
            if not pts: continue
            ax.scatter(
                [v.x for v in pts],
                [v.y for v in pts],
                [v.z for v in pts],
                color=color,
                s=100,
                label=base,
            )

        # plot the adjusted path
        ax.plot(
            [v.x for v in current],
            [v.y for v in current],
            [v.z for v in current],
            color="black",
            alpha=0.7,
            linewidth=2,
            label="Adjusted Path",
        )

        ax.set_title(f"Frame {frame_idx+1}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
        return []

    # ─── 7) Build & save animation ────────────────────────────────
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init,
        interval=interval, repeat=False
    )
    writer = animation.writers["ffmpeg"](fps=1000 // interval, metadata={"artist":"You"}, bitrate=1800)
    ani.save(save_path, writer=writer)
    plt.close(fig)

    print(f"Video saved to {save_path} in {(time.time() - start_time):.2f}s")
# ============= Compute Similarity =============
def compute_similarity(
    path_1: str, path_2: str, timeout: float = 3.0
) -> Optional[float]:
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
            ["../USalign/USalign", path_1, path_2], capture_output=True, text=True
        )
    except Exception as e:
        print(f"Error running USalign: {e}")
        return None

    stdout = result.stdout

    # regex to find the TM-score normalized by Structure_1
    m = re.search(
        r"TM-score=\s*([0-9]+(?:\.[0-9]+)?)\s*\(normalized by length of Structure_1",
        stdout,
    )
    if not m:
        print("TM-score (normalized by Structure_1) not found in USalign output.")
            
        return 0
    return float(m.group(1))

def correct_chirality(coords: List[Vector]) -> List[Vector]:
    """
    Detects if a backbone coordinate list is left-handed (mirrored) and corrects it by reflecting the x-axis.
    Returns a list of Vector objects in right-handed form.

    Parameters
    ----------
    coords : List[Vector]
        List of Vector instances representing the backbone in sequential order.

    Returns
    -------
    List[Vector]
        Coordinates converted to natural right-handed chirality.
    """
    # Convert to NumPy array (N, 3)
    arr = np.stack([v.as_array() for v in coords], axis=0)

    # Handedness undefined for fewer than 4 points
    if arr.shape[0] < 4:
        return coords

    # Compute backbone vectors: b_i = p_{i+1} - p_i
    b = arr[1:] - arr[:-1]  # shape (N-1, 3)

    # Compute signed triple products for i=0..N-4: (b[i] x b[i+1]) · b[i+2]
    # Use consistent slicing: b[:-2], b[1:-1], b[2:]
    triples = np.einsum("ij,ij->i", np.cross(b[:-2], b[1:-1]), b[2:])

    # Filter out exact zeros (collinear triples)
    non_zero = triples[triples != 0]

    # Determine average sign: positive = right-handed, negative = left-handed
    mean_sign = float(np.mean(np.sign(non_zero))) if non_zero.size else 0.0

    # Reflect x-axis if left-handed
    flip = False
    if mean_sign < 0:
        arr[:, 0] *= -1.0
        flip = True

    # Convert back to List[Vector]
    corrected = [Vector.from_array(row) for row in arr]
    return corrected, flip
