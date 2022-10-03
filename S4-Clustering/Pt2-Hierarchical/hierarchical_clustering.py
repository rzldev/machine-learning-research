## Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values

## Using the Dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

## Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_cluster = cluster.fit_predict(X)
print(y_cluster)

## Visualizing the clusters
plt.scatter(X[y_cluster == 0, 0], X[y_cluster == 0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X[y_cluster == 1, 0], X[y_cluster == 1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X[y_cluster == 2, 0], X[y_cluster == 2, 1], s=50, c='green', label='Cluster 3')
plt.scatter(X[y_cluster == 3, 0], X[y_cluster == 3, 1], s=50, c='brown', label='Cluster 4')
plt.scatter(X[y_cluster == 4, 0], X[y_cluster == 4, 1], s=50, c='magenta', label='Cluster 5')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
