from clusters import Evaluator, nucleotides_to_clusters,create_random_nucleotides, RealGenerator
from DataTypes import Cluster, Nucleotide, Vector
import random
import torch
import os

# ========== TESTING PARAMETERS ==========
TEST_SIZE = 5000
CLUSTER_SIZE = 4
K = 2 # noise range Angstrom

if __name__ == "__main__":
    # set file dir as current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # ------ RANDOM STRANDS TEST  ------
    
    # test accuracy on random clusters
    random_nucleotides = create_random_nucleotides(TEST_SIZE)
    
    random_clusters = nucleotides_to_clusters(
        random_nucleotides, real=False, cluster_size=CLUSTER_SIZE
    )
    real_generator = RealGenerator(cluster_size=CLUSTER_SIZE)
    real_clusters = real_generator.make_clusters(n_clusters=TEST_SIZE)
    all_clusters = random_clusters + real_clusters
    # random.shuffle(all_clusters)
    
    # load evaluator model
    evaluator = Evaluator(cluster_size=CLUSTER_SIZE)
    evaluator.load_state_dict(torch.load("models/evaluator.pt"))
    evaluator.eval()
    
    n_clusters = len(all_clusters)
    score = 0
    
    for cluster in all_clusters:
        pred = evaluator.eval_cluster(cluster)
        # print(f"Pred: {pred}, Real: {cluster.real}")
        if pred > 0.5:
            pred = True
        else:
            pred = False
        if cluster.real == pred:
            score += 1
            
    accuracy = score / n_clusters

    print(f"Number of clusters: {n_clusters}")
    print(f"Accuracy: {accuracy:.2f}")
    
    
    # ------ REAL STRAND + NOISE TEST ------
    print("---------- Harder Test ----------")
    real_generator = RealGenerator(cluster_size=CLUSTER_SIZE)
    real_clusters = real_generator.make_clusters(n_clusters=TEST_SIZE)
    noisy_clusters = real_generator.make_clusters(n_clusters=TEST_SIZE)

    for cluster in noisy_clusters:
        vectors = []
        for _ in range(CLUSTER_SIZE):
            dx = random.uniform(-K, K)
            dy = random.uniform(-K, K)
            dz = random.uniform(-K, K)
            vectors.append(Vector(x=dx, y=dy, z=dz))
        cluster.update(vectors)
        cluster.real = False    

    all_clusters = real_clusters + noisy_clusters
    # random.shuffle(all_clusters)
    # load evaluator model
    evaluator = Evaluator(cluster_size=CLUSTER_SIZE)
    evaluator.load_state_dict(torch.load("models/evaluator.pt"))
    evaluator.eval()
    n_clusters = len(all_clusters)
    score = 0
    for cluster in all_clusters:
        pred = evaluator.eval_cluster(cluster)
        
        # print(f"Pred: {pred}, Real: {cluster.real}")
        if pred > 0.5:
            pred = True
        else:
            pred = False
        if cluster.real == pred:
            score += 1
    accuracy = score / n_clusters
    print(f"Number of clusters: {n_clusters}")
    print(f"Accuracy: {accuracy:.2f}")
    
    
        
        