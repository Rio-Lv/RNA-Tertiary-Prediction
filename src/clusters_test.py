from clusters import (
    Evaluator,
    Adjuster,
    nucleotides_to_clusters,
    create_random_nucleotides,
    RealGenerator,
)
from DataTypes import Cluster, Nucleotide, Vector
import random
import torch
import os

# ========== TESTING PARAMETERS ==========
TEST_SIZE = 500
CLUSTER_SIZE = 6
K = 2  # noise range Angstrom

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
    print(random.choice(random_clusters))



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
    print(random.choice(real_clusters))

    
    # print("---------- Adjuster Test Simple ----------")
    # # Effect larger steps in nucleotides as a whole
    # # ------ Same Random + Adjustments test ------
    # # check if helps trick evaluator
    # adjuster = Adjuster(cluster_size=CLUSTER_SIZE, n_iter=1)
    # adjuster.load("models/adjuster.pt")
    # adjusted_nucleotides = random_nucleotides.copy()
    
    # # simpler: just update clusters then evaluate
    # # form clusters from nucleotides
    # adjusted_clusters = nucleotides_to_clusters(
    #     adjusted_nucleotides, real=False, cluster_size=CLUSTER_SIZE
    # )
    # for _ in range(1):
    #     # apply changes to clusters using adjuster
    #     for i in range(len(adjusted_clusters)):
    #         adjusted_clusters[i] = adjuster.update_cluster(cluster=adjusted_clusters[i])
    
    
    # all_clusters = adjusted_clusters + real_clusters
    # random.shuffle(all_clusters)
    
    # score = 0
    # for cluster in all_clusters:
    #     pred = evaluator.eval_cluster(cluster)

    #     # print(f"Pred: {pred}, Real: {cluster.real}")
    #     if pred > 0.5:
    #         pred = True
    #     else:
    #         pred = False
    #     if cluster.real == pred:
    #         score += 1
    # accuracy = score / n_clusters
    # print(f"Number of clusters: {n_clusters}")
    # print(f"Accuracy: {accuracy:.2f}")
    # print(random.choice(adjusted_clusters))
    
    
    # # Advanced: Update using vector differences on whole nucleotides list
    # for _ in range(6):
    #     # form clusters from nucleotides
    #     adjusted_clusters = nucleotides_to_clusters(
    #         adjusted_nucleotides, real=False, cluster_size=CLUSTER_SIZE
    #     )
    #     # apply changes to clusters using adjuster
    #     for i in range(len(adjusted_clusters)):
    #         adjusted_clusters[i] = adjuster.update_cluster(cluster=adjusted_clusters[i])
        
    #     adjusted_nucleotides = [adjusted_clusters[i].nucleotides[0] for i in range(len(adjusted_clusters))]
    
    
    print("---------- Adjuster Test Advanced ----------")
    # Effect larger steps in nucleotides as a whole
    # ------ Same Random + Adjustments test ------
    # check if helps trick evaluator
    adjuster = Adjuster(cluster_size=CLUSTER_SIZE, n_iter=1)
    adjuster.load("models/adjuster.pt")
    adjusted_nucleotides = random_nucleotides.copy()
    
    # simpler: just update clusters then evaluate
    # form clusters from nucleotides
    adjusted_clusters = nucleotides_to_clusters(
        adjusted_nucleotides, real=False, cluster_size=CLUSTER_SIZE
    )
    # Advanced: Update using vector differences on whole nucleotides list
    for _ in range(6):
        # apply changes to clusters using adjuster
        for i in range(len(adjusted_clusters)):
            adjusted_clusters[i] = adjuster.update_cluster(cluster=adjusted_clusters[i])
        
        # this need t obe updated to vector method
        adjusted_nucleotides = [adjusted_clusters[i].nucleotides[0] for i in range(len(adjusted_clusters))]
                # update clusters from new nucleotides
        adjusted_clusters = nucleotides_to_clusters(
            adjusted_nucleotides, real=False, cluster_size=CLUSTER_SIZE
        )
    
    
    all_clusters = adjusted_clusters + real_clusters
    random.shuffle(all_clusters)
    
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
    print(random.choice(adjusted_clusters))
    
    
   