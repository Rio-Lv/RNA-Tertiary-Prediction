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
            "all_sequences": [">1SCL_1|Chain A|RN...", ">1RNK_1|Chain A|RN...", ">1RHT_1|Chain A|RN..."],
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
        pdb += f"ATOM  {i+1:5}  CA  {strand.resname[i]:3} A{strand.resid[i]:4}    {strand.x_1[i]:8.3f}{strand.y_1[i]:8.3f}{strand.z_1[i]:8.3f}  1.00  0.00\n"
    return pdb


# ============= Save PDB =============
def save_pdb(pdb: str, filename: str):
    with open(filename, "w") as f:
        f.write(pdb)

# ============= Main/Test=============
if __name__ == "__main__":
    pdb_id = "2LKR_A"
    labels = Labels(df=pd.read_csv("data/train_labels.csv"))
    sequences = Sequences(df=pd.read_csv("data/train_sequences.csv"))
    # labels = Labels()
    # strand = Strand()
    strand = grab_strand(pdb_id, labels)
    sequence = grab_sequence(pdb_id, sequences)
    # print(labels)
    # print(strand)
    print(sequence)
    
    pdb = strand_to_pdb(strand)
    print(pdb)
    save_pdb(pdb, f"data/pdbs/tools_test_{pdb_id}.pdb")
    print("saved")
