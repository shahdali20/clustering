import numpy as np 
import pandas as pd 
from sklearn.cluster import DBSCAN
from functions import get_score_labels
from sklearn.decomposition import NMF
import plotly.express as px
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score as ss

start_time = time.time()
feature_vector1 = np.load('C:\\Users\\haider\\Desktop\\grad_poject\\code\\final1-VGG16\\feature_vector.npy')
#feature_vector2 = np.load('C:\\Users\\haider\\Desktop\\grad_poject\\code\\final1-VGG16\\feature_vector2.npy')
#scaler = StandardScaler()
#features1_ = scaler.fit_transform(feature_vector1)
#features2_ = scaler.fit_transform(feature_vector2)
nmf = NMF(n_components=3,max_iter=500)
features1_NMF = nmf.fit_transform(feature_vector1)
#features2_NMF = nmf.fit_transform(features2_)


#best_dect = get_score_labels(features1_NMF,'cosine')
#print(best_dect)

dbscan = DBSCAN(eps=0.0009478947368421053,min_samples=14,metric='cosine').fit(features1_NMF)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
labels = dbscan.labels_
dlad = pd.DataFrame(labels)
print(labels)
print(dlad.value_counts(normalize=False))
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(num_clusters)
print("score: ",ss(features1_NMF,labels,metric='cosine'))
fig = px.scatter(x=features1_NMF[:,0],y=features1_NMF[:,1],color=labels)
fig.show()


#dataset1
"""
{'best_eps': 0.0009478947368421053, 'best_min_samp': 14, 'best_labels': array([-1,  0,  1, ...,  2,  2,  2], dtype=int64), 'best_score': 0.63150734, 'best_clusters': 3, 'best_dflab':  1   
 910
 2    764
 0    634
-1    562
Name: count, dtype: int64}

"""