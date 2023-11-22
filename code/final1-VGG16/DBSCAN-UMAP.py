import numpy as np 
import pandas as pd 
from sklearn.cluster import DBSCAN
from functions import get_score_labels,KNN
import umap
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score as ss
from scipy.spatial.distance import euclidean,cosine
from sklearn.preprocessing import StandardScaler
import time

def calculate_rmse(actual, predicted):
    # Calculate RMSE
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def calculate_sse(actual, predicted):
    # Calculate SSE
    sse = np.sum((actual - predicted) ** 2)
    return sse

start_time = time.time()
#feature_vector1 = np.load('C:\\Users\\haider\\Desktop\\grad_poject\\code\\final1-VGG16\\feature_vector.npy')
feature_vector2 = np.load('C:\\Users\\haider\\Desktop\\grad_poject\\code\\final1-VGG16\\feature_vector2.npy')
#scaler = StandardScaler()
#features1_ = scaler.fit_transform(feature_vector1)
#features2_ = scaler.fit_transform(feature_vector2)
umap = umap.UMAP(n_components=3,min_dist=0.1, n_neighbors=15)
#features1_UMAP = umap.fit_transform(feature_vector1)
features2_UMAP = umap.fit_transform(feature_vector2)

#best_dect = get_score_labels(features2_UMAP,'cosine')
#print(best_dect)



dbscan = DBSCAN(eps=0.0001,min_samples=45,metric='cosine').fit(features2_UMAP)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

labels = dbscan.labels_
dlad = pd.DataFrame(labels)
print(labels)
print(dlad.value_counts(normalize=False))
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(num_clusters)
print("score: ",ss(features2_UMAP,labels,metric='cosine'))
fig = px.scatter(x=features2_UMAP[:,0],y=features2_UMAP[:,1],color=labels)
fig.show()


#dataset1
"""
ep = 0.001 / min = 14 
"""


#feature2 
#eps=0.0001,min_samples=45
#feature2 scaling
"""
{'best_eps': 0.056189213483146074, 'best_min_samp': 56, 'best_labels': array([0, 0, 0, ..., 3, 1, 3], dtype=int64), 'best_score': 0.5839896, 'best_clusters': 1, 'best_dflab':  0    2505
 1    2057
 2     449
 3     360
 4     179
-1      56}

"""