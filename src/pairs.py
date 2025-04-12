import os
import pandas as pd
from clusters import RealGenerator
# set file dir as current dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))

real_generator = RealGenerator(cluster_size=10)
real_clusters = real_generator.make_clusters(n_clusters=100)

for cluster in real_clusters[:10]:
    cluster_array = cluster.get_array()
    cols = ["dx", "dy", "dz", "a", "c", "g", "u", "cb"]
    df = pd.DataFrame(cluster_array, columns=cols)
    # add d column sqrt(x**2 + y**2 + z**2)
    df["d"] = df.apply(
        lambda row: (row["dx"] ** 2 + row["dy"] ** 2 + row["dz"] ** 2) ** 0.5, axis=1
    )
    print(df)
    
seq_df = pd.read_csv("data/train_sequences.csv")
sequences = seq_df["sequence"].tolist()
# get the shortest sequence
shortest_sequence = min(sequences, key=len)
print(f"Shortest sequence: {shortest_sequence}")
# 10 shortest sequences
shortest_sequences = sorted(sequences, key=len)[:10]
print(f"10 shortest sequences: {shortest_sequences}")
    
    