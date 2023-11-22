import numpy as np 
import pandas as pd 
from sklearn.cluster import DBSCAN
from functions import get_score_labels
from sklearn.decomposition import RandomProj


feature_vector = np.load('code\\final1-VGG16\\feature_vector.npy')
random_proj = RandomProj(n_components=100)
features_LSH = random_proj.fit_transform(feature_vector)

print(feature_vector.shape)
print(features_NMF.shape)

best_dect = get_score_labels(features_NMF,'cosine')
print(best_dect)

"""dbscan = DBSCAN(eps=0.63543,min_samples=1).fit(feature_vector)

labels = dbscan.labels_
dlad = pd.DataFrame(labels)
print(labels)
print(dlad.value_counts(normalize=False))
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(num_clusters)
"""

