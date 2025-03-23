import os
import pandas as pd
from pydantic import BaseModel, ConfigDict


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


# ============= Strand Class =============
class Strand(BaseModel):
    ID: list[str] = ["1SCL_A_1", "1SCL_A_2"]
    resname: list[str] = ["G", "G"]
    resid: list[int] = [1, 2]
    x_1: list[float] = [12.3, 15.6]
    y_1: list[float] = [7.8, 9.1]
    z_1: list[float] = [3.4, 4.5]


class Sequence(BaseModel):
    target_id: str = "1SCL_A"
    sequence: str = "GGGU"
    temporal_cutoff: str = "1995-01-26"
    description: str = "description1"
    all_sequences: str = ">1SCL_1|Chain A|RN..."


# ============= Grab Strand =============
def grab_strand(pdb_id: str, labels: Labels) -> Strand:
    filtered_df = labels.df[labels.df["ID"].str.contains(pdb_id)]

    return Strand(
        **{
            field: filtered_df[field].tolist()
            for field in Strand.model_fields
            if field in filtered_df.columns
        }
    )


def grab_sequence(pdb_id: str, sequences: Sequences) -> Sequence:
    filtered_df = sequences.df[sequences.df["target_id"] == pdb_id]
    return Sequence(
        **{
            field: filtered_df[field].tolist()[0]
            for field in Sequence.model_fields
            if field in filtered_df.columns
        }
    )


# ============= Strand to PDB =============
def strand_to_pdb(strand: Strand) -> str:
    pdb = ""
    for i in range(len(strand.ID)):
        # empty values are currently stored as -1e+18
        # if x y or z are < -1e+6, replace all 3 with space
        index = f"{i+1:5}"
        ca= "CA"
        resname = f"{strand.resname[i]:3}"
        resid = f"A{strand.resid[i]:4}"
        x_1 = strand.x_1[i]
        y_1 = strand.y_1[i]
        z_1 = strand.z_1[i]
   
        suffix = "1.00  0.00"
        
        if x_1 < -1e6 or y_1 < -1e6 or z_1 < -1e6:
            continue
        
        x_1 = f"{strand.x_1[i]:8.3f}"
        y_1 = f"{strand.y_1[i]:8.3f}"
        z_1 = f"{strand.z_1[i]:8.3f}"
        
        pdb += f"ATOM  {index}  {ca}  {resname} {resid}    {x_1} {y_1} {z_1} {suffix}\n"
    return pdb


# ============= Save PDB =============
def save_pdb(pdb: str, filename: str):
    with open(filename, "w") as f:
        f.write(pdb)


# ============= Compute Similarity =============
def compute_similarity(path_1: str, path_2: str):
    # use USalign to compute similarity
    os.system(f"USalign/USalign {path_1} {path_2}")


# ============= Main/Test=============
if __name__ == "__main__":
    # pdb_id = "2LKR_A" # exists in training_labels.csv
    pdb_id = "R1116"  # exists in validation_labels.csv
    labels = Labels(df=pd.read_csv("data/validation_labels.csv"))
    sequences = Sequences(df=pd.read_csv("data/validation_sequences.csv"))
    # labels = Labels()
    # strand = Strand()
    strand = grab_strand(pdb_id, labels)
    sequence = grab_sequence(pdb_id, sequences)
    # print(labels)
    # print(strand)
    print(sequence)

    pdb = strand_to_pdb(strand)
    print(pdb)
    path = f"data/pdbs/tools_test_{pdb_id}.pdb"
    save_pdb(pdb, path)
    print("saved to: ", path)

    path_1 = f"data/pdbs/strand_generator_simulation_2LKR_A.pdb"
    path_2 = f"data/pdbs/tools_test_R1116.pdb"
    compute_similarity(path_1, path_2)
