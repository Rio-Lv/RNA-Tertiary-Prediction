import pandas as pd
def grab_strand(pdb_id: str, df: pd.DataFrame) -> pd.DataFrame:
    # Ensure the 'ID' column is treated as a string and filter rows where search_id is a substring.
    filtered_df = df[df['ID'].astype(str).str.contains(pdb_id)]
    return filtered_df
