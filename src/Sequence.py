from typing import Literal
from torch import Tensor

res_types = Literal["A", "C", "G", "T"]

class Sequence:
    def __init__(self, str: str):
        self.encoding = self.encode_str(str)
        
    def __repr__(self):
        return f"Sequence({self.encoding})"
        
    @staticmethod
    def encode_str(str: str):
        """
        Convert a string to hot-encoded tensor.
        "A" -> [1, 0, 0, 0]
        "C" -> [0, 1, 0, 0]
        "G" -> [0, 0, 1, 0]
        "T" -> [0, 0, 0, 1]
        "N" -> [0, 0, 0, 0]
        Tensor([
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
            "T": [0, 0, 0, 1]
        }
        tensor = []
        for char in str:
            if char not in encoding:
                tensor.append([0, 0, 0, 0])
                continue
            tensor.append(encoding[char])
        assert len(tensor) == len(str), "Length of tensor does not match length of string"
        assert len(tensor[0]) == 4, "Length of tensor does not match length of encoding"
        return Tensor(tensor)

    @staticmethod
    def generate_coords(types: Tensor):
        """
        Generate coordinates for the sequence.
        """
        

if __name__ == "__main__":
    # Test the Sequence class
    seq = Sequence("ACGTAA8")
    print(seq)