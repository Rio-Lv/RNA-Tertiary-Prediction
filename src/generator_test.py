from generator import Evaluator, nucleotides_to_clusters, RealGenerator
from DataTypes import Cluster, Nucleotide, Vector
import random
import torch
import os

if __name__ == "__main__":
    # set file dir as current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # test accuracy on random clusters
    test_size = 1000
    cluster_size = 5
    
    random_nucleotides = [
        Nucleotide(
            index=i,
            type=random.choice(["A", "C", "G", "U", "N"]),
            coordinate=Vector(
                x=random.uniform(-10, 10),
                y=random.uniform(-10, 10),
                z=random.uniform(-10, 10),
            ),
        )
        for i in range(test_size)
    ]
    
    random_clusters = nucleotides_to_clusters(
        random_nucleotides, real=False, cluster_size=cluster_size
    )
    real_generator = RealGenerator(cluster_size=cluster_size)
    real_clusters = real_generator.make_clusters(n_clusters=test_size)
    all_clusters = random_clusters + real_clusters
    random.shuffle(all_clusters)
    
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
    
    print("---------- Harder Test ----------")
    
    # Harder Test where you use real clusters + some noise
    test_size = 1000
    cluster_size = 5
    real_generator = RealGenerator(cluster_size=cluster_size)
    real_clusters = real_generator.make_clusters(n_clusters=test_size)
    noisy_clusters = real_generator.make_clusters(n_clusters=test_size)
    k = 10
    for cluster in noisy_clusters:
        cluster.real = False    
        for nucleotide in cluster.nucleotides:
            nucleotide.coordinate.x += random.uniform(-k, k)
            nucleotide.coordinate.y += random.uniform(-k, k)
            nucleotide.coordinate.z += random.uniform(-k, k)

    all_clusters = real_clusters + noisy_clusters
    random.shuffle(all_clusters)
    # load evaluator model
    evaluator = Evaluator(cluster_size=cluster_size)
    evaluator.load_state_dict(torch.load("models_pretty_good/evaluator.pt"))
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
    
    
        
        