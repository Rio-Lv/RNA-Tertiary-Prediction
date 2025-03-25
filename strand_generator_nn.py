from tools import *
from pydantic import BaseModel

import torch


class Generator:
    nucleotides: list[Nucleotide]
    input_shape: tuple[int] = (
        5,
        3 + 4,
    )  # base nucleotide and 4 nearest neigbors, 3 coordinates + 4 nucleotide types mapped to 4 values
    output_shape: tuple[int] = (3,)  # 3 coordinates for displacement of base nucleotide
    distance_matrix: torch.Tensor
    model: torch.nn.Module
    type_map: dict[str, list[int]] = {
        "A": [1, 0, 0, 0],
        "U": [0, 1, 0, 0],
        "C": [0, 0, 1, 0],
        "G": [0, 0, 0, 1],
    }

    def __init__(self, nucleotides: list[Nucleotide]):
        self.nucleotides = nucleotides
        self.update_distance_matrix()
        self.init_model()

    def init_model(self):
        input_shape = self.input_shape
        output_shape = self.output_shape
        model = torch.nn.Sequential(
            torch.nn.Linear(input_shape[0] * input_shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_shape[0]),
        )
        self.model = model

    def update_distance_matrix(self):
        distance_matrix = torch.zeros(len(self.nucleotides), len(self.nucleotides))
        for i, nt in enumerate(self.nucleotides):
            for j, nt2 in enumerate(self.nucleotides):
                dx = nt.coordinate.x - nt2.coordinate.x
                dy = nt.coordinate.y - nt2.coordinate.y
                dz = nt.coordinate.z - nt2.coordinate.z
                distance = (dx**2 + dy**2 + dz**2) ** 0.5
                distance_matrix[i][j] = distance
        self.distance_matrix = distance_matrix

    def generate_input(self, base_index: int) -> torch.Tensor:
        # get 4 nearest neighbors
        distance_matrix = self.distance_matrix
        nearest = torch.argsort(distance_matrix[base_index])
        nearest = nearest[1:5]
        # get input indices (base + 4 nearest)
        input_indices = [base_index] + nearest.tolist()

        # loop through nucleotides and create input tensor
        type_map = self.type_map
        inputs = []
        for i in input_indices:
            nt = self.nucleotides[i]
            inputs.append(
                [nt.coordinate.x, nt.coordinate.y, nt.coordinate.z] + type_map[nt.type]
            )

        return torch.tensor(inputs)


# =================== Model Creation ===================

# =================== Model Training ===================

# =================== Model Validation ===================
if __name__ == "__main__":
    print("============ Strand Generator Neural Network ===========")
    print("--------------------------------------------------------")
    print("1. Checking for MPS Device")
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")
    print("--------------------------------------------------------")
    print("2. Getting Random Sequence for Model Testing")
    # 1.1. Load Sequences and Labels
    labels_path = "data/validation_labels.csv"
    sequences_path = "data/validation_sequences.csv"
    labels = Labels(df=pd.read_csv(labels_path))
    sequences = Sequences(df=pd.read_csv(sequences_path))
    print("labels_path:", labels_path)
    print("sequences_path:", sequences_path)

    # 1.2. Grab Random Sequence
    sequence = grab_random_sequence(sequences)
    nucleotides = sequence_to_nucleotide_line(sequence)

    print("--------------------------------------------------------")
    print("3. Initialising Generator")
    generator = Generator(nucleotides=nucleotides)
    print(generator.distance_matrix)
    print(generator.model)
    
    print("--------------------------------------------------------")
    print("4. Input Tensor:")
    input_tensor = generator.generate_input(0)
    print(input_tensor)
