import itertools
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score as ss
from sklearn.neighbors import NearestNeighbors


# A function to try several combanitions of epsilons and min samples
# to determine the best epsilon and min sample 
def get_score_labels(X,metric):
    scores = []
    all_labels = []
    
    epsilon = np.linspace(0.001,0.5,num=20)
    min_samples = np.arange(1,100,step=1)
    comb = list(itertools.product(epsilon,min_samples))
    N = len(comb)
    for i,(eps,num_samples) in enumerate(comb):
        dbscan = DBSCAN(eps=eps, min_samples=num_samples,metric=metric).fit(X)
        labels = dbscan.labels_
        labels_set = set(labels)
        num_clusters = len(labels_set)
        
        if -1 in labels_set:
            num_clusters -= 1
        if (num_clusters < 3) or (num_clusters > 5):
            scores.append(-10)
            all_labels.append("bad")
            c = (eps,num_samples)
            print(f"combinations {c} on iteration {i+1} of {N} has {num_clusters} clusters. moving on")
            continue
        
        scores.append(ss(X,labels,metric='cosine'))
        all_labels.append(labels)
        dlad = pd.DataFrame(labels)
        print(f"index: {i}, score: {scores[-1]}, labels: {all_labels[-1]}, number of cluster: {num_clusters}, labels count: \n {dlad.value_counts(normalize=False)}")
        
    best_index = np.argmax(scores)
    print("best index: ",best_index)
    best_parameters = comb[best_index]
    best_labels = all_labels[best_index]
    best_scores = scores[best_index]
    best_clusters = num_clusters
    dlad = pd.DataFrame(best_labels)
    lab_count = dlad.value_counts(normalize=False)
    return {
        "best_eps": best_parameters[0],
        "best_min_samp": best_parameters[1],
        "best_labels": best_labels,
        "best_score": best_scores,
        "best_clusters": best_clusters,
        "best_dflab": lab_count
    }


# A function to calculate the optimal epsilon using KNN
def KNN(X,min_samples):
    k = min_samples
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    average_distances = distances.mean(axis=1)
    sorted_distances = np.sort(average_distances)
    diffs = np.diff(sorted_distances)
    elbow_index = np.argmax(diffs)
    optimal_epsilon = sorted_distances[elbow_index]
    return optimal_epsilon