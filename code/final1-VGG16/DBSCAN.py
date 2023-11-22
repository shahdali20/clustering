import numpy as np 
import pandas as pd 
from sklearn.cluster import DBSCAN
from functions import get_score_labels
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score as ss


start_time = time.time()
feature_vector1 = np.load('code\\final1-VGG16\\feature_vector.npy')
#feature_vector2 = np.load('code\\final1-VGG16\\feature_vector2.npy')


#best_dect = get_score_labels(feature_vector1,'cosine')
#print(best_dect)


dbscan = DBSCAN(eps=0.4212105263157895,min_samples=6,metric='cosine').fit(feature_vector1)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
labels = dbscan.labels_
print(labels)
dlad = pd.DataFrame(labels)
print(dlad.value_counts(normalize=False))
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(num_clusters)
print("score: ",ss(feature_vector1,labels,metric='cosine'))
#fig = px.scatter(x=feature_vector1[:,0],y=feature_vector1[:,1],color=labels)
#fig.show()

"""{'best_eps': 0.8333499999999999, 'best_min_samp': 67, 'best_labels': array([ 0,  0, -1, ...,  1,  1,  1], dtype=int64), 'best_score': 0.05700758, 'best_clusters': 1, 'best_dflab':  0    952
-1    891
 1    704
 2    323
 
 
 
 {'best_eps': 0.4212105263157895, 'best_min_samp': 6, 'best_labels': array([0, 0, 1, ..., 1, 1, 1], dtype=int64), 'best_score': 0.1042201, 'best_clusters': 1, 'best_dflab':  1    1618
 0     961
-1     284
 2       7
Name: count, dtype: int64}

"""