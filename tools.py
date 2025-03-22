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


# ============= Strand Class =============
class Strand(BaseModel):
    ID: list[str] = ["1SCL_A_1", "1SCL_A_2"]
    resname: list[str] = ["G", "G"]
    resid: list[int] = [1, 2]
    x_1: list[float] = [12.3, 15.6]
    y_1: list[float] = [7.8, 9.1]
    z_1: list[float] = [3.4, 4.5]


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

# ============= Test =============
def default_test():
    labels = Labels(df=pd.read_csv("data/train_labels.csv"))
    # labels = Labels()
    # strand = Strand()
    strand = grab_strand("2LKR_A", labels)
    print(labels)
    print(strand)
    pdb = strand_to_pdb(strand)
    print(pdb)
    save_pdb(pdb, "data/pdbs/test.pdb")
    print("saved")
# ============= Main =============
if __name__ == "__main__":
    default_test()
