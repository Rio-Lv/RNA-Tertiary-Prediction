from torch import Tensor
from typing import Literal
from torch import Tensor
import random
import pandas as pd


class Vector:
    x: float
    y: float
    z: float

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vector({self.x}, {self.y}, {self.z})"


class Nucleotide:
    index: int
    type: Literal["A", "C", "G", "U", "N"]
    coordinate: Vector
    array: list[float]
    neighbors: list["Nucleotide"]

    def __init__(
        self,
        index: int = 0,
        type: Literal["A", "C", "G", "U", "N"] = "N",
        coordinate: Vector = Vector(),
        neighbors: list["Nucleotide"] = [],
    ):
        self.index = index
        self.type = type
        self.coordinate = coordinate
        self.array = self.get_array()
        self.neighbors = neighbors
        assert (
            len(self.array) == 7
        ), f"Nucleotide array must be of length 7. Got {len(self.array)}"

    def __repr__(self):
        return f"Nucleotide({self.index}, {self.type}, {self.coordinate})"

    def get_coord_array(self):
        return [self.coordinate.x, self.coordinate.y, self.coordinate.z]

    def get_type_array(self):
        if self.type == "A":
            return [1, 0, 0, 0]
        elif self.type == "C":
            return [0, 1, 0, 0]
        elif self.type == "G":
            return [0, 0, 1, 0]
        elif self.type == "U":
            return [0, 0, 0, 1]
        else:
            return [0, 0, 0, 0]

    def get_array(self):
        tensor = [self.coordinate.x, self.coordinate.y, self.coordinate.z]
        tensor += self.get_type_array()
        return tensor


class Cluster:
    """
    Coord Vectors should be relative to the first nucleotide in the cluster.
    """

    nucleotides: list[Nucleotide]
    array: list[list[float]]  # [dx, dy, dz, a, c, g, u, connected_to_base]
    tensor: Tensor

    def __init__(
        self,
        real: bool,
        nucleotides: list[Nucleotide],
        cluster_size: int
        
    ):
        self.real = real
        self.nucleotides = nucleotides
        self.array = self.get_array()
        self.tensor = self.get_tensor()
        self.cluster_size = cluster_size

        assert self.tensor.shape == (
            cluster_size,
            8,
        ), f"Cluster tensor must be of shape ({cluster_size}, 8). Got {self.tensor.shape}, cluster: {self}"

    def __repr__(self):
        """
        Cluster representation.
        """
        columns = ["dx", "dy", "dz", "A", "C", "G", "U", "CB", "real"]
        # Create a string representation of the cluster

        def format_val(val, width):
            # Convert from np.float64 to float if necessary
            if isinstance(val, Tensor):
                val = val.item()
            # For floats, try fixed-point with 2 decimals first.
            if isinstance(val, float):
                s = f"{val:{width}.2f}"
                if len(s) <= width:
                    return s
                # If the fixed-point format is too long, try scientific notation with 1 decimal.
                s = f"{val:{width}.1e}"
                if len(s) <= width:
                    return s
                # Fallback: truncate to the specified width.
                return s[:width]
            else:
                # For integers (or other types), format as an integer.
                s = f"{val:{width}d}"
                if len(s) <= width:
                    return s
                return s[:width]

        # Build the header with specific widths.
        header_parts = []
        for i, col in enumerate(columns):
            if i < 3:
                header_parts.append(f"{col:>5}")
            else:
                header_parts.append(f"{col:>3}")
        header = "    " + " ".join(header_parts) 

        # Build the rows, applying the correct width for each column.
        rows = [header]
        for row in self.array:
            formatted_row_parts = []
            for i, val in enumerate(row):
                width = 5 if i < 3 else 3
                formatted_row_parts.append(format_val(val, width))
            # add real value
            real_val = 1 if self.real else 0
            formatted_row_parts.append(format_val(real_val, 3))
            formatted_row = "    " + " ".join(formatted_row_parts)
            rows.append(formatted_row)

        rows.append("    " + "-" * 47)
        # add line break

        array_str = "\n".join(rows)
        return array_str

    def get_array(self):
        base_nucleotide = self.nucleotides[0]
        relative_nucleotides = []
        connected_to_base = []
        base_index = base_nucleotide.index

        for nucleotide in self.nucleotides:
            relative_nucleotide = Nucleotide(
                index=nucleotide.index,
                type=nucleotide.type,
                coordinate=Vector(
                    x=nucleotide.coordinate.x - base_nucleotide.coordinate.x,
                    y=nucleotide.coordinate.y - base_nucleotide.coordinate.y,
                    z=nucleotide.coordinate.z - base_nucleotide.coordinate.z,
                ),
            )
            relative_nucleotides.append(relative_nucleotide)

            # Check if the nucleotide is connected to the base nucleotide
            if abs(nucleotide.index - base_index) == 1:
                connected_to_base.append(1)
            else:
                connected_to_base.append(0)

        array = [nucleotide.get_array() for nucleotide in relative_nucleotides]
        array = [row + [connected_to_base[i]] for i, row in enumerate(array)]
        return array

    def get_tensor(self):
        return Tensor(self.array)

    def update(self, vectors: list[Vector]):
        """
        Move the base nucleotide by a given delta vector.
        """
        assert len(vectors) == len(self.nucleotides), (
            f"Vectors length {len(vectors)} must be equal to nucleotides length {len(self.nucleotides)}"
        )
        for i in range(len(self.nucleotides)):
            nucleotide = self.nucleotides[i]
            vector = vectors[i]
            dx = vector.x
            dy = vector.y
            dz = vector.z
            nucleotide.coordinate.x += dx
            nucleotide.coordinate.y += dy
            nucleotide.coordinate.z += dz
            
        self.array = self.get_array()
        self.tensor = self.get_tensor()


if __name__ == "__main__":
    # test_vector = Vector(x=1.0, y=2.0, z=3.0)
    # print(test_vector)
    # test_nucleotide = Nucleotide(index=1, type="A", coordinate=test_vector)
    # print(test_nucleotide)
    cluster_size = 10
    test_cluster = Cluster(
        real=False,
        nucleotides=[
            Nucleotide(
                i,
                random.choice(["A", "C", "G", "U", "N"]),
                Vector(
                    x=i * 1.0,
                    y=i * 2.0,
                    z=i * 3.0,
                ),
            )
            for i in range(cluster_size)
        ],
        cluster_size=cluster_size,
    )
    print(test_cluster)
    vectors = [Vector(x=i, y=1.0, z=1.0) for i in range(cluster_size)]
    test_cluster.update(vectors)
    print(test_cluster)
