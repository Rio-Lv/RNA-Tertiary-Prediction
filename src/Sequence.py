import os
from typing import Optional
from torch import Tensor
from torch import nn
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import copy
import math
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

# set here to cwd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ====== CONSTANTS ======
CLUSTER_SIZE = 100
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
        return (Vector(self.x, self.y, self.z),)

    def add(self, vector: "Vector"):
        self.x += vector.x
        self.y += vector.y
        self.z += vector.z


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

    def __init__(self, seq_str: str, seq_id: str = None, coords: list[Vector] = None):
        self.seq_str = seq_str
        self.seq_id = seq_id
        self.encoding = self.encode_str(seq_str)
        self.coords = (
            copy.deepcopy(coords)
            if coords
            else self._coords_to_noise(walk=True)
        )
        self.source_coords = copy.deepcopy(self.coords)
        self.distance_matrix = self.compute_distance_matrix(self.coords)

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

    def adjust_coords(self, k: float = 0.05, n_iter: int = 100, heat:float = 5) -> list[Vector]:
        """
        Make coords match the distance matrix. Via Simulation.
        1. Calculate distance matrix from current coordinates
        2. Calc difference between distance matrix and target distance matrix
        3. Create list of dx, dy, dz for each coordinate
        4. Calc all unit vectors from coord i to coord j

        """
        for _ in range(n_iter):
            new_distance_matrix = self.compute_distance_matrix(self.coords)
            diff_mat = new_distance_matrix - self.distance_matrix
            # adjust coordinates based on diff
            n_coords = len(self.coords)
            deltas: list[Vector] = [Vector(0, 0, 0) for _ in range(n_coords)]
            for i in range(len(self.coords)):
                for j in range(len(self.coords)):
                    if i == j:
                        continue
                    dx = self.coords[j].x - self.coords[i].x
                    dy = self.coords[j].y - self.coords[i].y
                    dz = self.coords[j].z - self.coords[i].z
                    dist = Sequence.distance(self.coords[i], self.coords[j]) + random.uniform(-heat, heat)
                    diff = diff_mat[i][j]
                    ux = dx / dist
                    uy = dy / dist
                    uz = dz / dist
                    delta = Vector(ux * diff * k, uy * diff * k, uz * diff * k)
                    deltas[i].add(delta)

            for i in range(len(self.coords)):
                self.coords[i].x += deltas[i].x
                self.coords[i].y += deltas[i].y
                self.coords[i].z += deltas[i].z
        return self.coords

    def to_pdb(self, save_path: str = None) -> str:
        pdb_str = ""
        for i in range(len(self.coords)):
            x = self.coords[i].x
            y = self.coords[i].y
            z = self.coords[i].z
            resname = self.seq_str[i]
            pdb_str += f"ATOM  {i+1:5d}  CA  {resname} A{1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
        pdb_str += "END\n"
        if save_path:
            with open(save_path, "w") as f:
                f.write(pdb_str)
            print(f"Saved PDB to {save_path}")
        return pdb_str

    def align(self, target_coords):
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
        if len(self.coords) < 3 or len(target_coords) < 3:
            raise ValueError(
                "At least 3 coordinates are required in both the source and target sets."
            )

        # --- Step 1: Define planes for source and target using the first three points ---
        # Source plane
        p0 = np.array([self.coords[0].x, self.coords[0].y, self.coords[0].z])
        p1 = np.array([self.coords[1].x, self.coords[1].y, self.coords[1].z])
        p2 = np.array([self.coords[2].x, self.coords[2].y, self.coords[2].z])
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
        points = np.array([[vec.x, vec.y, vec.z] for vec in self.coords])
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
        self.coords = new_coords
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
        cmap = cm.get_cmap("tab10", len(coords_list))

        # Iterate over each vector set and plot the points and connecting line.
        for i, coords in enumerate(coords_list):
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
    def _coords_to_noise(self, walk:bool = True, noise_k: float = 5):
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
        iterations:int,
        step_k:float,
        video_filename="adjustment.mp4",
        interval=33,
    ):
        """
        Adjust the coordinates to match the distance matrix and output a video
        showing the adjustment process over multiple iterations.

        The original coordinates are shown in gray (with both scatter points
        and a connecting line) while the adjusted coordinates are plotted in red.

        Parameters:
        video_filename (str): The filename of the output video.
        iterations (int): Total number of adjustment iterations (default 200).
        interval (int): Delay between frames in milliseconds (default ~33 ms for 30fps).
        """
        print("Starting video generation...")

        # Record the starting time.
        start_time = time.time()

        # Deep copy the original coordinates.
        original_coords = self.source_coords  # Assuming self.source_coords exists.

        # Add random noise to the coordinates.
        self._coords_to_noise()

        # Compute the bounding box based on the original coordinates with 25% padding.
        x_orig_vals = [coord.x for coord in original_coords]
        y_orig_vals = [coord.y for coord in original_coords]
        z_orig_vals = [coord.z for coord in original_coords]

        x_min, x_max = min(x_orig_vals), max(x_orig_vals)
        y_min, y_max = min(y_orig_vals), max(y_orig_vals)
        z_min, z_max = min(z_orig_vals), max(z_orig_vals)

        # Compute ranges and add 25% padding.
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        # If a range is 0, assign a small default value.
        if x_range == 0: x_range = 1.0
        if y_range == 0: y_range = 1.0
        if z_range == 0: z_range = 1.0

        x_pad = 0.25 * x_range
        y_pad = 0.25 * y_range
        z_pad = 0.25 * z_range

        x_lim = (x_min - x_pad, x_max + x_pad)
        y_lim = (y_min - y_pad, y_max + y_pad)
        z_lim = (z_min - z_pad, z_max + z_pad)

        # Create a 3D plotting figure.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        def init():
            ax.clear()
            # Fix the axis limits.
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            # Plot the original coordinates as points.
            x_orig = [coord.x for coord in original_coords]
            y_orig = [coord.y for coord in original_coords]
            z_orig = [coord.z for coord in original_coords]
            ax.scatter(x_orig, y_orig, z_orig, color="gray", s=100, label="Original")
            # Plot a gray line connecting the original coordinates.
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
            # Print progress every 10 iterations.
            if (frame + 1) % 10 == 0 or frame == 0:
                print(f"Processing iteration {frame+1}/{iterations}")

            # Perform a single adjustment iteration.
            self.coords = self.adjust_coords(k=step_k, n_iter=1)
            self.coords = self.align(original_coords)

            # Extract adjusted coordinates.
            x_adj = [coord.x for coord in self.coords]
            y_adj = [coord.y for coord in self.coords]
            z_adj = [coord.z for coord in self.coords]

            # Clear and replot. Then set fixed axis limits.
            ax.clear()
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)

            # Original (static)
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
            # Adjusted (dynamic)
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
            ax.set_title(f"Adjustment Iteration {frame+1}")
            ax.legend(loc="upper right")
            return []

        # Create the animation object.
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(iterations),
            init_func=init,
            interval=interval,
            repeat=False,
        )

        # Save the animation to a video file using the FFmpeg writer.
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=1000 // interval, metadata=dict(artist="Your Name"), bitrate=1800)
        ani.save(video_filename, writer=writer)
        plt.close(fig)

        elapsed_time = time.time() - start_time
        print(f"Video saved to {video_filename} in {elapsed_time:.2f} seconds")

        self.plot([original_coords, self.coords], ["Original", "Adjusted"])

# ====== DATA PPEPERATION ======
class SequenceDataset:
    """
    Inputs is the Sequence Encoding
    Output is the distance matrix
    """

    real_sequences: list[Sequence]

    def __init__(self, n_sequences: int = 1000):
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

    def get_random_sequence(self):
        seq = random.choice(self.real_sequences)
        while len(seq.coords) > CLUSTER_SIZE+20 and len(seq.coords) < CLUSTER_SIZE-20:
            seq = random.choice(self.real_sequences)
        print(seq)
        return seq


# ======== MODELS ==========
class DistanceMatrixModel(nn.Module):
    """
    Takes in a sequence and outputs a distance matrix.
    """

    def __init__():
        super().__init__()


if __name__ == "__main__":

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

    # =============== Test 2 ==============
    seq_dataset = SequenceDataset()
    print(len(seq_dataset.real_sequences))
    # Initialize a real sequence (Distance Matrix Assigned)
    seq = seq_dataset.get_random_sequence()
    seq._test_adjust_coords_video(iterations=500,step_k=0.01)
