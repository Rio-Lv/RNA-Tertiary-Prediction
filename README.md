# RNA-Tertiary-Prediction

# Manifesting Rank 1 Stanford Kaggle Competition on RNA Tertiary Structure Predictions

Current Thought Process:
1. Understand the data
- it seems relatively simple. 
- train_labels.csv has the coords of in order of sequence eg the C-1 atom of A in ACGUAG... 
- train.csv has the sequence of the RNA in ACGUAG... 

2. First Attempt:
- input should be nucleotide and 3 nearest neighbors []
- output should be dx, dy, dz of the C-1 atom of the nucleotide


3. Data Sources:
- https://www.rcsb.org/  - has the 3D structure of the RNA