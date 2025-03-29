# ============= Imports =============
import os
import pandas as pd
from pydantic import BaseModel, ConfigDict
import random
import time
from typing import Literal, Tuple
import math
import numpy as np
from numpy import ndarray
from torch import Tensor
from typing import Optional
from torch.utils.data import Dataset

# ============= Coordinate Class =============
class Coordinate(BaseModel):
    x: float
    y: float
    z: float

# ============= Nucleotide Class =============
class Nucleotide(BaseModel):
    index: int
    type: Literal["A", "C", "G", "U" , "-"]
    coordinate: Coordinate = Coordinate(x=0, y=0, z=0)
    is_fake: Optional[bool] = None
    
    def get_array(self):
        return [
                self.coordinate.x,
                self.coordinate.y,
                self.coordinate.z,
                1 if self.type == "A" else 0,
                1 if self.type == "C" else 0,
                1 if self.type == "G" else 0,
                1 if self.type == "U" else 0,
                1 if self.type == "-" else 0,
            ]

# ============= Labels Class =============
class Labels(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    df: pd.DataFrame = pd.DataFrame(
        {
            "ID": ["1SCL_A_1", "1SCL_A_2", "8Z1F_T_1"],
            "resname": ["G", "G", "G"],
            "resid": [1, 2, 1],
            "x_1": [12.3, 15.6, 18.9],
            "y_1": [7.8, 9.1, 10.4],
            "z_1": [3.4, 4.5, 5.6],
        }
    )


# ============= Strand Class =============
class Strand(BaseModel):
    ID: list[str] = ["1SCL_A_1", "1SCL_A_2"]
    resname: list[str] = ["G", "G"]
    resid: list[int] = [1, 2]
    x_1: list[float] = [12.3, 15.6]
    y_1: list[float] = [7.8, 9.1]
    z_1: list[float] = [3.4, 4.5]
    
    def summarise(self):
        # return readable table
        print("----- Strand Summary -----")
        df = pd.DataFrame(
            {
                "ID": self.ID,
                "resname": self.resname,
                "resid": self.resid,
                "x_1": self.x_1,
                "y_1": self.y_1,
                "z_1": self.z_1,
            }
        )
        print(df)
        

# ============= Sequences Class =============
class Sequences(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    df: pd.DataFrame = pd.DataFrame(
        {
            "target_id": ["1SCL_A", "1RNK_A", "1RHT_A"],
            "sequence": ["GGGU", "GGGA", "GAGU"],
            "temporal_cutoff": ["1995-01-26", "1995-01-26", "1995-01-26"],
            "description": ["description1", "description2", "description3"],
            "all_sequences": [
                ">1SCL_1|Chain A|RN...",
                ">1RNK_1|Chain A|RN...",
                ">1RHT_1|Chain A|RN...",
            ],
        }
    )


# ============= Sequence Class =============
class Sequence(BaseModel):
    target_id: str = "1SCL_A"
    sequence: str = "GGGU"
    temporal_cutoff: str = "1995-01-26"
    description: str = "description1"
    all_sequences: str = ">1SCL_1|Chain A|RN..."


# ============= Evaluator Dataset Class =============
class EvaluatorDataset(Dataset):
    def __init__(self, fake_clusters: list[Tensor], real_clusters: list[Tensor]):
        
        is_fake = []
        clusters = []
        clusters_removed = 0
        
        for i in range(len(fake_clusters)):
            # check both do not contain NaN values
            if not np.isnan(fake_clusters[i]).any() and not np.isnan(real_clusters[i]).any():
                clusters.append(fake_clusters[i])
                is_fake.append(0)
                clusters.append(real_clusters[i])
                is_fake.append(1)
            else:
                clusters_removed += 1
        self.clusters = clusters
        self.is_fake = is_fake
        self.clusters_removed = clusters_removed
        
        assert fake_clusters[0].shape == (5, 8), f"Fake Cluster should be of shape (5, 8) not {fake_clusters[0].shape}"
        assert real_clusters[0].shape == (5, 8), f"Real Cluster should be of shape (5, 8) not {real_clusters[0].shape}"
        assert len(fake_clusters) == len(real_clusters), "Fake and Real Clusters should be the same length"
        assert len(fake_clusters) > 0, "Fake Clusters should not be empty"
        assert len(real_clusters) > 0, "Real Clusters should not be empty"
        assert len(fake_clusters[0]) == 5, "Fake Cluster should have 5 nucleotides"
        assert len(real_clusters[0]) == 5, "Real Cluster should have 5 nucleotides"
        
        

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, idx):
        return self.clusters[idx], self.is_fake[idx]
    
    def summarise(self):
        print("----- Evaluator Dataset Summary -----")
        print(f"Number of samples: {len(self)}")
        print(f"Number of clusters: {len(self.clusters)}")
        print(f"Number of fake clusters: {self.is_fake.count(0)}")
        print(f"Number of real clusters: {self.is_fake.count(1)}")
        print(f"Shape of clusters: {self.clusters[0].shape}")
        print(f"Clusters removed: {self.clusters_removed}")
        # create preview table
        # Create preview table by flattening each cluster to 35 values.
        flattened_clusters = [cluster.flatten().tolist() for cluster in self.clusters[:5]]
        shape = self.clusters[0].shape
        flat_length = shape[0] * shape[1]
        
        preview_df = pd.DataFrame(flattened_clusters, columns=[f"val_{i}" for i in range(flat_length)])
        preview_df["is_fake"] = self.is_fake[:5]
        print("Preview of first 5 clusters (flattened):")
        print(preview_df)

# ============= Grab Strand =============
def grab_strand(pdb_id: str, labels: Labels) -> Strand:
    filtered_df = labels.df[labels.df["ID"].str.contains(pdb_id)]

    filtered_df = filtered_df.dropna(subset=["x_1", "y_1", "z_1"])
    return Strand(
        **{
            field: filtered_df[field].tolist()
            for field in Strand.model_fields
            if field in filtered_df.columns
        }
    )


# ============= Grab Sequence by ID =============
def grab_sequence(pdb_id: str, sequences: Sequences) -> Sequence:
    filtered_df = sequences.df[sequences.df["target_id"] == pdb_id]
    # drop na values
    filtered_df = filtered_df.dropna(subset=["sequence"])
    return Sequence(
        **{
            field: filtered_df[field].tolist()[0]
            for field in Sequence.model_fields
            if field in filtered_df.columns
        }
    )


# ============= Grab Random Sequence =============
def grab_random_sequence(sequences: Sequences) -> Sequence:
    sequences_length = len(sequences.df)
    random_index = random.randint(0, sequences_length)
    return Sequence(
        **{
            field: sequences.df[field].tolist()[random_index]
            for field in Sequence.model_fields
            if field in sequences.df.columns
        }
    )


# ============= Sequence to Nucleotide Line =============
def sequence_to_nucleotide_line(
    sequence: Sequence
) -> list[Nucleotide]:
    # nucleotides: list[Nucleotide] = [
    #     Nucleotide(index=i, type=nt, coordinate=Coordinate(x=0, y=0, z=i * 6.5))
    #     for i, nt in enumerate(sequence.sequence)
    # ]
    nucleotides = []
    for i, nt in enumerate(sequence.sequence):
        # Calculate coordinates based on index
        x = i * 6.5
        y = random.uniform(-0.5, 0.5)
        z = random.uniform(-0.5, 0.5)
        # check if nt is valid NT 
        if nt in ["A", "C", "G", "U", "-"]:
            nucleotides.append(
                Nucleotide(
                    index=i,
                    type=nt,
                    coordinate=Coordinate(x=x, y=y, z=z),
                )
            )

    return nucleotides


# ============= Strand to Nucleotides =============
def strand_to_nucleotides(strand: Strand) -> list[Nucleotide]:
    nucleotides = [
        Nucleotide(
            index=i,
            type=strand.resname[i],
            coordinate=Coordinate(
                x=strand.x_1[i],
                y=strand.y_1[i],
                z=strand.z_1[i],
            ),
        )
        for i in range(len(strand.ID))
    ]
    return nucleotides


# ============= Nucleotides to Strand =============
def nucleotides_to_strand(
    nucleotides: list[Nucleotide], id="generated_strand"
) -> Strand:
    strand = Strand(
        ID=[id + f"_{nt.index}" for nt in nucleotides],
        resname=[nt.type for nt in nucleotides],
        resid=[nt.index for nt in nucleotides],
        x_1=[nt.coordinate.x for nt in nucleotides],
        y_1=[nt.coordinate.y for nt in nucleotides],
        z_1=[nt.coordinate.z for nt in nucleotides],
    )
    return strand


# ============= Rotate Nucleotides =============
def rotate_nucleotides(
    nucleotides: list[Nucleotide], rx: float = 0, ry: float = 0, rz: float = 0
) -> list[Nucleotide]:
    """
    Rotate nucleotides around their centroid using extrinsic rotations in x, y, z order.
    Angles are expected in degrees.
    """
    # Convert degrees to radians
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)
    

    # Calculate centroid
    cx = sum(nt.coordinate.x for nt in nucleotides) / len(nucleotides)
    cy = sum(nt.coordinate.y for nt in nucleotides) / len(nucleotides)
    cz = sum(nt.coordinate.z for nt in nucleotides) / len(nucleotides)

    # Precompute trigonometric values
    cos_x, sin_x = math.cos(rx), math.sin(rx)
    cos_y, sin_y = math.cos(ry), math.sin(ry)
    cos_z, sin_z = math.cos(rz), math.sin(rz)

    # Combined rotation matrix elements (extrinsic: Rz * Ry * Rx)
    # Derived from matrix multiplication of Rz @ Ry @ Rx
    m00 = cos_z * cos_y
    m01 = cos_z * sin_y * sin_x - sin_z * cos_x
    m02 = cos_z * sin_y * cos_x + sin_z * sin_x
    m10 = sin_z * cos_y
    m11 = sin_z * sin_y * sin_x + cos_z * cos_x
    m12 = sin_z * sin_y * cos_x - cos_z * sin_x
    m20 = -sin_y
    m21 = cos_y * sin_x
    m22 = cos_y * cos_x

    # Apply combined rotation to each nucleotide
    for nt in nucleotides:
        # check if coordinates are valid
        if (
            nt.coordinate.x < -1e10
            or nt.coordinate.y < -1e10
            or nt.coordinate.z < -1e10
            or math.isnan(nt.coordinate.x)
            or math.isnan(nt.coordinate.y)
            or math.isnan(nt.coordinate.z)
        ):
            continue
        # Translate to centroid origin
        x = nt.coordinate.x - cx
        y = nt.coordinate.y - cy
        z = nt.coordinate.z - cz

        # Apply rotation matrix
        new_x = x * m00 + y * m01 + z * m02
        new_y = x * m10 + y * m11 + z * m12
        new_z = x * m20 + y * m21 + z * m22

        # Translate back
        nt.coordinate.x = new_x + cx
        nt.coordinate.y = new_y + cy
        nt.coordinate.z = new_z + cz
    # Return rotated nucleotides
    return nucleotides


# ============ Get Neighboring Nucleotides =============
# 2. Get N nearest Nucleotides
def get_nearest_nucleotides(
    N: int, index: int, nucleotides: list[Nucleotide], distance_matrix
) -> list[Nucleotide]:
    distances = distance_matrix[index]
    nearest = np.argsort(distances)
    return [nucleotides[i] for i in nearest[0:N]]


# ============= Calculate Distance Matrix =============
def calculate_distance_matrix(nucleotides: list[Nucleotide]) -> ndarray:
    matrix = []
    for nt in nucleotides:
        distances = []
        for nt2 in nucleotides:
            distance = (
                (nt.coordinate.x - nt2.coordinate.x) ** 2
                + (nt.coordinate.y - nt2.coordinate.y) ** 2
                + (nt.coordinate.z - nt2.coordinate.z) ** 2
            ) ** 0.5
            distances.append(distance)
        matrix.append(distances)
    return np.array(matrix)


# ============= Contract Nucleotides =============
def contract_nucleotides(nucleotides: list[Nucleotide]):
    """_summary_
    if the distance between two nucleotides is not 6.5, displace whole strand

    Args:
        nucleotides (list[Nucleotide]): _description_
    """
    for i in range(1, len(nucleotides)):
        nt1 = nucleotides[i - 1]
        nt2 = nucleotides[i]
        distance = (
            (nt1.coordinate.x - nt2.coordinate.x) ** 2
            + (nt1.coordinate.y - nt2.coordinate.y) ** 2
            + (nt1.coordinate.z - nt2.coordinate.z) ** 2
        ) ** 0.5
        d = max(distance - 6.5, 0.1)
        
        
        ux = (nt2.coordinate.x - nt1.coordinate.x) / distance
        uy = (nt2.coordinate.y - nt1.coordinate.y) / distance
        uz = (nt2.coordinate.z - nt1.coordinate.z) / distance
        dx = ux * d
        dy = uy * d
        dz = uz * d
        for nt in nucleotides[i:]:
            nt.coordinate.x -= dx
            nt.coordinate.y -= dy
            nt.coordinate.z -= dz


# ============= Strand to PDB =============
def strand_to_pdb(strand: Strand) -> str:
    pdb = ""
    for i in range(len(strand.ID)):
        # Skip if coordinates are missing (using -1e18 as per your comment)
        if (
            strand.x_1[i] < -1e10
            or strand.y_1[i] < -1e10
            or strand.z_1[i] < -1e10
            or math.isnan(strand.x_1[i])
            or math.isnan(strand.y_1[i])
            or math.isnan(strand.z_1[i])
        ):
            continue
        
        resname = strand.resname[i]
        if resname == "-":
            resname = "N" # Unknown nucleotide

        # Format fields according to PDB standard
        pdb_line = (
            f"ATOM  {i+1:5}  CA {strand.resname[i]:>3} A{strand.resid[i]:4}    "
            f"{strand.x_1[i]:8.3f}{strand.y_1[i]:8.3f}{strand.z_1[i]:8.3f}"
            f"  1.00  0.00"
        )
        pdb += pdb_line + "\n"

    return pdb


# ============= Save PDB =============
def save_pdb(pdb: str, filename: str):
    with open(filename, "w") as f:
        f.write(pdb)


# ============= Compute Similarity =============
def compute_similarity(path_1: str, path_2: str):
    # use USalign to compute similarity
    # loop check if file exists max 3s
    start = time.time()
    while not os.path.exists(path_1) or not os.path.exists(path_2):
        if time.time() - start > 3:
            break
    if os.path.exists(path_1) and os.path.exists(path_2):
        os.system(f"../USalign/USalign {path_1} {path_2}")
    else:
        print("Files not found")
    return


# ============= Main/Test=============
if __name__ == "__main__":
    # set dir to file location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # 1. choose data paths
    # sequences_path = "data/validation_sequences.csv"
    # labels_path = "data/validation_labels.csv"
    sequences_path = "data/train_sequences.csv"
    labels_path = "data/train_labels.csv"

    # 2. init sequences and labels
    sequences = Sequences(df=pd.read_csv(sequences_path))
    labels = Labels(df=pd.read_csv(labels_path))

    # 3. grab random sequence and strand
    sequence = grab_random_sequence(sequences)
    pdb_id = sequence.target_id
    # # 3.1 optionally, grab a specific strand
    pdb_id = "4V6W_A5" # Big One 2000+ Working now after df drop na! 
    # pdb_id = "4V4N_A1" # Big One 2000+ Working
    # pdb_id = "6WB1_C" # had non ACGU

    strand = grab_strand(pdb_id, labels)
    strand.summarise()
   
    # 4. convert strand to nucleotides
    nucleotides = strand_to_nucleotides(strand)
    print("step 4: nucleotides")
    # 5. rotate nucleotides
    nucleotides_rotated = rotate_nucleotides(nucleotides, rx=0, ry=0, rz=90)

    # 6. convert nucleotides to strand
    strand_rotated = nucleotides_to_strand(nucleotides_rotated)

    # 7. convert both strands to pdb
    pdb_strand = strand_to_pdb(strand)
    pdb_strand_rotated = strand_to_pdb(strand_rotated)

    # 8. save both pdb files
    save_path_strand = f"data/tools_test/{pdb_id}_strand.pdb"
    save_path_strand_rotated = f"data/tools_test/{pdb_id}_strand_rotated.pdb"
    save_pdb(pdb_strand, save_path_strand)
    save_pdb(pdb_strand_rotated, save_path_strand_rotated)

    # 9. compute similarity
    compute_similarity(save_path_strand, save_path_strand_rotated)
    print("Completed Generation and Comparison for", pdb_id)
