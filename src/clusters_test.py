from clusters import Evaluator, nucleotides_to_clusters,create_random_nucleotides, RealGenerator
from DataTypes import Cluster, Nucleotide, Vector
import random
import torch
import os

if __name__ == "__main__":
    # set file dir as current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # ------ PRIMARY PARAMS ------
    test_size = 3000
    cluster_size = 5
    
    # ------ RANDOM STRANDS TEST  ------
    
    # test accuracy on random clusters
    random_nucleotides = create_random_nucleotides(test_size)
    
    random_clusters = nucleotides_to_clusters(
        random_nucleotides, real=False, cluster_size=cluster_size
    )
    real_generator = RealGenerator(cluster_size=cluster_size)
    real_clusters = real_generator.make_clusters(n_clusters=test_size)
    all_clusters = random_clusters + real_clusters
    # random.shuffle(all_clusters)
    
    # load evaluator model
    evaluator = Evaluator(cluster_size=cluster_size)
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
    real_generator = RealGenerator(cluster_size=cluster_size)
    real_clusters = real_generator.make_clusters(n_clusters=test_size)
    noisy_clusters = real_generator.make_clusters(n_clusters=test_size)
    k = 2
    for cluster in noisy_clusters:
        vectors = []
        for _ in range(cluster_size):
            dx = random.uniform(-k, k)
            dy = random.uniform(-k, k)
            dz = random.uniform(-k, k)
            vectors.append(Vector(x=dx, y=dy, z=dz))
        cluster.update(vectors)
        cluster.real = False    

    all_clusters = real_clusters + noisy_clusters
    # random.shuffle(all_clusters)
    # load evaluator model
    evaluator = Evaluator(cluster_size=cluster_size)
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
    
    
        
        