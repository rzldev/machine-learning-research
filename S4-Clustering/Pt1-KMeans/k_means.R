## K-Means

## Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

## Using the Elbow Method to find the optimal number of clusters
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(X, i)$withinss)
plot(1:10, wcss, type = 'b', 
     main = paste('The Elbow Method'), 
     xlab = 'Number of Clusters', ylab = 'WCSS')

## Applying KMeans model to the dataset
set.seed(29)
kmeans = kmeans(x = X, centers = 5, iter.max = 300, nstart = 10)

## Visualising the clusters
library(cluster)
clusplot(X, kmeans$cluster, 
         lines = 0, shade = TRUE, color = TRUE, labels = 2, plotchar = FALSE, span = TRUE,
         main = 'Clusters of Customers', xlab = 'Annual Income', ylab = 'Spending Score')
