import numpy as np 
import pandas as pd 
from sklearn.cluster import DBSCAN
from functions import get_score_labels,KNN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score as ss
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

start_time = time.time()
feature_vector1 = np.load('C:\\Users\\haider\\Desktop\\grad_poject\\code\\final1-VGG16\\feature_vector.npy')
#feature_vector2 = np.load('C:\\Users\\haider\\Desktop\\grad_poject\\code\\final1-VGG16\\feature_vector2.npy')
#scaler = StandardScaler()
#features1_ = scaler.fit_transform(feature_vector1)
#features2_ = scaler.fit_transform(feature_vector2)
pca = PCA(n_components=3)
features1_PCA = pca.fit_transform(feature_vector1)
#features2_PCA = pca.fit_transform(features2_)


#best_dect = get_score_labels(features2_PCA,'cosine')
#print(best_dect)


dbscan = DBSCAN(eps= 0.0211,min_samples=86,metric='cosine').fit(features1_PCA)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
labels = dbscan.labels_
dlad = pd.DataFrame(labels)
print(labels)
print(ss(features1_PCA,labels,metric='cosine'))
print(dlad.value_counts(normalize=False))
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(num_clusters)
print("score: ",ss(features1_PCA,labels,metric='cosine'))
fig = px.scatter(x=features1_PCA[:,0],y=features1_PCA[:,1],color=labels)
fig.show()




#dataset1 
"""
0.0211 // 86
{'best_eps': 0.083425, 'best_min_samp': 55, 'best_labels': array([0, 0, 1, ..., 3, 3, 3], dtype=int64), 'best_score': 0.44286552, 'best_clusters': 1, 'best_dflab':  0    998
 1    882
 2    488
 3    285
-1    217
Name: count, dtype: int64}

{'best_eps': 0.083425, 'best_min_samp': 59, 'best_labels': array([0, 0, 1, ..., 4, 4, 4], dtype=int64), 'best_score': 0.51164126, 'best_clusters': 1, 'best_dflab':  0    991
 3    488
 2    483
 1    393
 4    282
-1    233
Name: count, dtype: int64}
"""

