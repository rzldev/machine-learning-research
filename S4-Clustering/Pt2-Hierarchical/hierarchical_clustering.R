## Hierarchical Clustering

## Importing the dataset
dataset = read.csv('Mall_Customers.csv')
dataset = dataset[4:5]

## Using the Dendrogram to find the optimal the number of clusters
dendrogram = hclust(dist(dataset, method = 'euclidean'), method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean Distance')

## Fitting Hierarchical Clustering model on the dataset
hc = hclust(dist(dataset, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

## Visualizing the clusters
library(cluster)
clusplot(dataset, y_hc,
         lines = 0, shade = TRUE, color = TRUE, labels = 2, plotchar = FALSE, span = TRUE,
         main = 'Clusters of Customers', xlab = 'Annual Income', ylab = 'Spending Score')
