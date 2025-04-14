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
from tools import compute_similarity

# set here to cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ====== CONSTANTS ======
SEQUENCE_SIZE = 100
# N_NEAREST_NEIGBORS = 30  # If using n nearest neighbors for adjustment
MAX_DISTANCE = 12  # If using neightbor within distance for adjustment
USE_NEIGHBORS = False  # If using n nearest neighbors for adjustment

ITERATIONS = 20000
TEMPERATURE = 0.3
MAX_DELTA = 0.005
LABELS_PATH = "data/train_labels.csv"
SEQUENCES_PATH = "data/train_sequences.csv"
SEQUENCE_INDEX = 868

MAX_SPINE_SPACE = 7.7  # Maximum distance between two points in the spine

OPEN_PLOT = True  # If True, will open a plot window for each sequence
GRAVITY = 0.01
VIDEO_SPEED = 50  # Speed of the video in frames per second

# NOISY_SOURCE_MATRIX = True


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
    coords_matrix: Tensor

    def __init__(self, seq_str: str, seq_id: str = None, coords: list[Vector] = None):
        self.seq_str = seq_str
        self.seq_id = seq_id
        self.encoding = self.encode_str(seq_str)
        self.coords = (
            copy.deepcopy(coords) if coords else self._coords_to_noise(walk=True)
        )
        self.source_coords = copy.deepcopy(self.coords)
        self.distance_matrix = self.compute_distance_matrix(self.coords)
        self.coords_matrix = self.create_coords_matrix(self.coords)

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
    def create_coords_matrix(coords: list[Vector]) -> Tensor:
        """
        Create a tensor from a list of coordinates.
        :param coords: List of coordinates
        :return: Tensor of coordinates
        """
        coords_matrix = []
        for coord in coords:
            coords_matrix.append([coord.x, coord.y, coord.z])
        return Tensor(coords_matrix)

    @staticmethod
    def compute_neighbors_list(
        coord_list: list[Vector], n_neighbors: int
    ) -> list[list[int]]:
        """
        Compute the n nearest neighbors for each coordinate.

        For each coordinate in coord_list, determine the indices of the n closest points.
        The output is a matrix of size (len(coord_list), n_neighbors) that stores the indices
        of the nearest neighbors for each coordinate.

        Parameters:
        coord_list (list[Vector]): List of coordinate vectors.
        n_neighbors (int): Number of nearest neighbors to find for each coordinate.

        Returns:
        Tensor: A 2D tensor (or list of lists) of nearest neighbor indices.
        """
        # First, compute the full distance matrix.
        distance_matrix = Sequence.compute_distance_matrix(coord_list)

        neighbors = []  # This will be a list of lists holding indices.
        n_points = len(coord_list)

        # For each point, find the indices corresponding to the n smallest distances
        # (ignoring the diagonal entry which is zero).
        for i in range(n_points):
            # Create a list of indices with their associated distance,
            # skipping the self-distance at index i.
            distances = [(j, distance_matrix[i][j]) for j in range(n_points) if j != i]
            # Sort the list by distance.
            distances.sort(key=lambda tup: tup[1])
            # Extract the indices of the n closest points.
            nearest_indices = [idx for idx, dist in distances[:n_neighbors]]
            neighbors.append(nearest_indices)

        return neighbors

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

    def correct_spine(self):
        """
        sometimes adjustments make spine coords too far
        so we need to correct them.
        1. loop through i and i+1
        2. if distance is greater than 7A
        3. calculate the difference vector
        4. normalize the vector
        5. multiply by 7A
        6. move [i+1:] by the difference vector
        7. return the new coords
        """
        for i in range(len(self.coords) - 1):
            dist = Sequence.distance(self.coords[i], self.coords[i + 1])
            if dist > MAX_SPINE_SPACE:
                dx = self.coords[i + 1].x - self.coords[i].x
                dy = self.coords[i + 1].y - self.coords[i].y
                dz = self.coords[i + 1].z - self.coords[i].z
                ux = dx / dist
                uy = dy / dist
                uz = dz / dist
                d = MAX_SPINE_SPACE - dist
                # Move the next points by the difference vector
                for j in range(i + 1, len(self.coords)):
                    self.coords[j].x += ux * d
                    self.coords[j].y += uy * d
                    self.coords[j].z += uz * d
        return self.coords

    def gravitate_centroid(self):
        """
        Move All Coordinates Very slightly towards the centroid.
        1. calculate centroid
        2. calculate the difference vector
        3. calculate unit vector towards centroid
        4. move each coord by the difference vector
        """
        centroid = Vector(0, 0, 0)
        # for coord in self.coords:
        #     centroid.x += coord.x
        #     centroid.y += coord.y
        #     centroid.z += coord.z
        # centroid.x /= len(self.coords)
        # centroid.y /= len(self.coords)
        # centroid.z /= len(self.coords)
        # Calculate the difference vector
        for coord in self.coords:
            dx = centroid.x - coord.x
            dy = centroid.y - coord.y
            dz = centroid.z - coord.z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            ux = dx / dist
            uy = dy / dist
            uz = dz / dist
            # Move the coord by the difference vector
            coord.x += ux * GRAVITY
            coord.y += uy * GRAVITY
            coord.z += uz * GRAVITY
        return self.coords

    def _adjust_coords_via_max_dist(self, n_iter: int) -> list[Vector]:
        """
        Make coords match the distance matrix via simulation.
        1. Calculate distance matrix from current coordinates.
        2. Calculate difference between the current and target distance matrices.
        3. Create list of adjustments for each coordinate.
        4. Calculate unit vectors from coord i to coord j.
        Only contributions from pairs with distances <= MAX_DISTANCE are considered.
        """
        coords_matrix = self.create_coords_matrix(self.coords)
        recording = []

        for curr in range(n_iter):
            print(f"Iteration {curr+1}/{n_iter}")
            # Compute current pairwise distances: shape [N, N]
            new_distance_matrix = torch.norm(
                coords_matrix.unsqueeze(0) - coords_matrix.unsqueeze(1), dim=2
            )

            # Compute difference between current distances and the original ones.
            dist_diff = new_distance_matrix - self.distance_matrix

            # Add Gaussian noise scaled by the absolute distance difference.
            heat = torch.normal(0, TEMPERATURE, size=dist_diff.shape)
            heat = torch.abs(dist_diff) * heat
            new_distance_matrix = new_distance_matrix + heat

            # Create a binary mask: 1 for distances within MAX_DISTANCE, 0 otherwise.
            mask = (new_distance_matrix <= MAX_DISTANCE).float()

            # Apply the mask to the distance differences so contributions outside the threshold vanish.
            dist_diff = dist_diff * mask

            # Compute pairwise coordinate differences: shape [N, N, 3]
            d_coords = coords_matrix.unsqueeze(0) - coords_matrix.unsqueeze(1)

            eps = 1e-8
            # Compute unit directional vectors.
            u_vecs = d_coords / (new_distance_matrix.unsqueeze(2) + eps)
            # Multiply unit vectors by the mask so that pairs outside the threshold contribute zero.
            u_vecs = u_vecs * mask.unsqueeze(2)

            # Compute the deltas by summing the contributions for each coordinate.
            deltas = (u_vecs * dist_diff.unsqueeze(2)).sum(dim=1)

            # Enforce that each delta's magnitude does not exceed MAX_DELTA.
            mags = torch.norm(deltas, dim=1, keepdim=True)
            scale = MAX_DELTA / (mags + eps)
            deltas = deltas * scale

            # Update the coordinates.
            coords_matrix = coords_matrix + deltas

            # Record the current state.
            recording.append(coords_matrix.clone())

        # Update self.coords from the coords_matrix.
        for i in range(len(self.coords)):
            self.coords[i] = Vector(coords_matrix[i][0], coords_matrix[i][1], coords_matrix[i][2])

        return self.coords, recording

    def adjust_coords(
        self, n_iter: int = 100, use_neighbors: bool = False
    ) -> tuple[list[Vector], list[Tensor]]:
        """
        Adjust the coordinates to match the distance matrix.
        Can use cluster size or max distance to adjust.
        """
        return self._adjust_coords_via_max_dist(n_iter)

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
        gen_path = "seq_output/seq_generated.pdb",
        target_path = "seq_output/seq_target.pdb"  
    ):

        compute_similarity(gen_path, target_path)

    @staticmethod
    def align(input_coords, target_coords):
        """
        1. Use the first 3 coordinates to create a plane for self.coords and target_coords.
        2. Create a quaternion from the planes using SciPy to align the plane normal of self.coords
        to that of target_coords.
        3. Rotate self.coords by the quaternion.
        4. Translate all points so that self.coords[0] aligns with target_coords[0].
        5. Align the unit vector from self.coords[0] to self.coords[1] with the unit vector from
        target_coords[0] to target_coords[1] (apply an additional twist rotation about the plane normal).
        6. Return the new coordinates as a list of Vector objects.

        Parameters:
        target_coords (list[Vector]): List of target Vector objects.

        Returns:
        list[Vector]: A new list of Vector objects representing the aligned coordinates.
        """
        # Ensure there are at least 3 points in both sets.
        if len(input_coords) < 3 or len(target_coords) < 3:
            raise ValueError(
                "At least 3 coordinates are required in both the source and target sets."
            )

        # --- Step 1: Define planes for source and target using the first three points ---
        # Source plane
        p0 = np.array([input_coords[0].x, input_coords[0].y, input_coords[0].z])
        p1 = np.array([input_coords[1].x, input_coords[1].y, input_coords[1].z])
        p2 = np.array([input_coords[2].x, input_coords[2].y, input_coords[2].z])
        v1 = p1 - p0
        v2 = p2 - p0
        n_source = np.cross(v1, v2)
        n_source_norm = n_source / np.linalg.norm(n_source)

        # Target plane
        q0 = np.array([target_coords[0].x, target_coords[0].y, target_coords[0].z])
        q1 = np.array([target_coords[1].x, target_coords[1].y, target_coords[1].z])
        q2 = np.array([target_coords[2].x, target_coords[2].y, target_coords[2].z])
        w1 = q1 - q0
        w2 = q2 - q0
        n_target = np.cross(w1, w2)
        n_target_norm = n_target / np.linalg.norm(n_target)

        # --- Step 2: Create rotation to align source normal to target normal ---
        # Note: align_vectors expects target first.
        rot_obj, rmsd = R.align_vectors([n_target_norm], [n_source_norm])

        # --- Step 3: Rotate all source points using the computed rotation ---
        points = np.array([[vec.x, vec.y, vec.z] for vec in input_coords])
        rotated_points = rot_obj.apply(points)

        # --- Step 4: Translate so that the first points align ---
        translation = q0 - rotated_points[0]
        aligned_points = rotated_points + translation

        # --- Step 5: Additional twist alignment to match the first-to-second point direction ---
        # Compute the unit vector from point 0 to point 1 in the source (after rotation & translation)
        vec_source = aligned_points[1] - aligned_points[0]
        d_source = vec_source / np.linalg.norm(vec_source)
        # And for the target:
        vec_target = q1 - q0
        d_target = vec_target / np.linalg.norm(vec_target)

        # Compute the angle between the directions.
        dot_val = np.clip(np.dot(d_source, d_target), -1.0, 1.0)
        angle = np.arccos(dot_val)
        # Determine the sign of the angle using the target plane normal as the reference axis.
        cross_vec = np.cross(d_source, d_target)
        sign = np.sign(np.dot(cross_vec, n_target_norm))
        twist_angle = angle * sign

        # Create twist rotation about the axis (which is n_target_norm)
        twist_rot = R.from_rotvec(twist_angle * n_target_norm)
        # Apply the twist rotation about the common pivot q0 (target_coords[0]).
        final_aligned_points = []
        for pt in aligned_points:
            final_pt = q0 + twist_rot.apply(pt - q0)
            final_aligned_points.append(final_pt)

        # --- Step 6: Convert back into a list of Vector objects and update self.coords ---
        new_coords = [Vector(pt[0], pt[1], pt[2]) for pt in final_aligned_points]
        # self.coords = new_coords
        return new_coords

    def plot(self, coords_list: list[list[Vector]] = None, set_names: list[str] = None):
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

        # If no coordinate sets are provided, use self.coords as a single vector set.
        if coords_list is None:
            coords_list = [self.coords]

        coords_list = [self.align(coords, self.source_coords) for coords in coords_list]

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
            coords = []
            curr_pos = Vector(
                random.uniform(-r, r),
                random.uniform(-r, r),
                random.uniform(-r, r),
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

    def _test_adjust_coords(self):
        """
        Try get structure from noise using distance matrix.
        1. Recreate coordinates with random noise
        2. Adjust coordinates to match distance matrix
        3. Plot the original and adjusted coordinates.
        """
        # Deep copy the original coordinates.
        original_coords = copy.deepcopy(self.coords)
        noise = 10
        self.coords = [
            Vector(
                random.uniform(-noise, noise),
                random.uniform(-noise, noise),
                random.uniform(-noise, noise),
            )
            for _ in self.coords
        ]
        self.adjust_coords(k=0.05, n_iter=100)
        self.plot([original_coords, self.coords], ["OG", "Adjusted"])
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
        coords_adjusted, recording = self.adjust_coords(
            n_iter=iterations, use_neighbors=USE_NEIGHBORS
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
            current_coords = self.align(recording_coords[frame], original_coords)
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
            self.plot([original_coords, self.coords], ["Original", "Adjusted"])


# ====== DATA PPEPERATION ======
class SequenceDataset:
    """
    Inputs is the Sequence Encoding
    Output is the distance matrix
    """

    real_sequences: list[Sequence]

    def __init__(self, n_sequences: int = 100):
        self.real_sequences = self.get_real_sequences(n_sequences)

    def get_real_sequences(self, target_n_sequences: int):
        label_df = pd.read_csv(LABELS_PATH)
        # using a windowed approach
        sequences = []

        max_length = len(label_df)
        curr_index = 0

        while (
            len(sequences) < target_n_sequences
            or curr_index + SEQUENCE_SIZE > max_length
        ):
            i = curr_index

            resid = label_df.iloc[i]["resid"]
            end_resid = label_df.iloc[i + SEQUENCE_SIZE]["resid"]
            if resid > end_resid:
                curr_index += SEQUENCE_SIZE
                continue
            labels_window = label_df.iloc[i : i + SEQUENCE_SIZE]
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
        # [print(seq) for seq in sequences[:5]]
        return sequences

    def get_random_sequence(self):
        random_index = random.randint(0, len(self.real_sequences) - 1)
        seq = self.real_sequences[random_index]
        curr_try = 0
        max_tries = 2000
        while (
            len(seq.coords) > SEQUENCE_SIZE + 20
            and len(seq.coords) < SEQUENCE_SIZE - 20
        ):
            random_index = random.randint(0, len(self.real_sequences) - 1)
            seq = self.real_sequences[random_index]
            curr_try += 1
            if curr_try > max_tries:
                print("Max tries reached, returning random sequence.")
                break
        seq = copy.deepcopy(seq)
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
class DistanceMatrixModel(nn.Module):
    """
    Takes in a sequence and outputs a distance matrix.
    """

    def __init__():
        super().__init__()


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
    # print(len(seq_dataset.real_sequences))
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

    # # =============== Test 2 ==============
    print("Loading Sequence Dataset")
    seq_dataset = SequenceDataset()
    print(len(seq_dataset.real_sequences))
    # Initialize a real sequence (Distance Matrix Assigned)
    # seq_dataset.get_stats()
    print("Loading Random Sequence")
    seq = seq_dataset.get_random_sequence()
    # seq = seq_dataset.real_sequences[SEQUENCE_INDEX]
    print(f"Sequence Length: {len(seq.seq_str)}")
    # seq._coords_to_noise()
    # seq.adjust_coords(n_iter=100, use_neighbors=USE_NEIGHBORS)
    seq._test_adjust_coords_video(iterations=ITERATIONS, speed=VIDEO_SPEED)
    
    Sequence.compute_similarity_us_align()
