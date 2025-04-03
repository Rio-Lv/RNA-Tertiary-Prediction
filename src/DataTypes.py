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
        ), f"Nucleotide array must be of length 8. Got {len(self.array)}"

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

    source_nucleotides: list[Nucleotide]
    array: list[list[float]]  # [dx, dy, dz, a, c, g, u, connected_to_base]
    tensor: Tensor

    def __init__(
        self, nucleotides: list[Nucleotide] = [Nucleotide() for _ in range(5)]
    ):
        assert len(nucleotides) == 5, "Cluster must contain exactly 5 nucleotides."
        self.source_nucleotides = nucleotides
        self.array = self.get_array()
        self.tensor = self.get_tensor()
        assert (
            len(self.array) == 5
        ), f"Cluster array must be of length 5. Got {len(self.array)}"
        assert self.tensor.shape == (
            5,
            8,
        ), f"Cluster tensor must be of shape (5, 7). Got {self.tensor.shape}"

    def __repr__(self):
        """
        Cluster representation.
        """

        columns = ["dx", "dy", "dz", "A", "C", "G", "U", "CB"]

        # Build the header with the specified widths
        header_parts = []
        for i, col in enumerate(columns):
            if i < 3:
                header_parts.append(f"{col:>5}")
            else:
                header_parts.append(f"{col:>2}")
        header = "    " + " ".join(header_parts)

        # Build the rows, formatting each value according to its column
        rows = []
        for row in self.array:
            formatted_row_parts = []
            for i, val in enumerate(row):
                if i < 3:
                    formatted_row_parts.append(f"{val:>5}")
                else:
                    formatted_row_parts.append(f"{val:>2}")
            formatted_row = "    " + " ".join(formatted_row_parts)
            rows.append(formatted_row)
        array_str = "\n".join(rows)

        return (
            f"Cluster(\n"
            f"    Number Of Nucleotides: {len(self.source_nucleotides)}\n"
            f"{header}\n"
            f"{array_str}\n"
            f")"
        )

    def get_array(self):
        base_nucleotide = self.source_nucleotides[0]
        relative_nucleotides = []
        connected_to_base = []
        base_index = base_nucleotide.index

        for nucleotide in self.source_nucleotides:
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

    def update(self, vector: Vector = Vector()):
        """
        Move the base nucleotide by a given delta vector.
        """
        dx = vector.x
        dy = vector.y
        dz = vector.z
        self.source_nucleotides[0].coordinate.x += dx
        self.source_nucleotides[0].coordinate.y += dy
        self.source_nucleotides[0].coordinate.z += dz
        self.array = self.get_array()
        self.tensor = self.get_tensor()
        
class Sequence:
    """
    A sequence of nucleotides.
    """

    nucleotides: list[Nucleotide]
    delaunays

    def __init__(self, nucleotides: list[Nucleotide] = []):
        self.nucleotides = nucleotides

    def __repr__(self):
        return f"Sequence({self.nucleotides})"

    def get_tensor(self):
        return Tensor([nucleotide.get_array() for nucleotide in self.nucleotides])


if __name__ == "__main__":
    test_vector = Vector(x=1.0, y=2.0, z=3.0)
    print(test_vector)
    test_nucleotide = Nucleotide(index=1, type="A", coordinate=test_vector)
    print(test_nucleotide)
    test_cluster = Cluster(
        nucleotides=[
            Nucleotide(
                i,
                random.choice(["A", "C", "G", "U"]),
                Vector(
                    x=i * 1.0,
                    y=i * 2.0,
                    z=i * 3.0,
                ),
            )
            for i in range(5)
        ]
    )
    print(test_cluster)
    vector = Vector(x=3.0, y=1.0, z=2.0)
    test_cluster.update(vector)
    print(test_cluster)
