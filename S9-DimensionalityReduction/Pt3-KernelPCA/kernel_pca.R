## Kernel PCA

## Importing the dataset
dataset = read.csv('Wine.csv')

## Splitting the dataset into the training set and test set
#install.packages('caTools')
library(caTools)
split = sample.split(dataset$Customer_Segment, SplitRatio = .8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

## Feature Scaling
training_set[-14] = scale(training_set[-14])
test_set[-14] = scale(test_set[-14])

## Applying the Kernel PCA
#install.packages('kernlab')
library(kernlab)
kpca = kpca(~., data = training_set[-3], kernel = 'rbfdot', features = 2)
training_set_kpca = as.data.frame(predict(kpca, training_set))
training_set_kpca$Customer_Segment = training_set$Customer_Segment
test_set_kpca = as.data.frame(predict(kpca, test_set))
test_set_kpca$Customer_Segment = test_set$Customer_Segment

## Training the SVM model on the training set
#install.packages('e1071')
library(e1071)
classifier = svm(formula = Customer_Segment ~ .,
                 data = training_set_kpca,
                 type = 'C-classification',
                 kernel = 'linear')

## Predicting the test set results
y_pred = predict(classifier, newdata = test_set_kpca[-3])

## Making the Confusion Matrix
cm = table(test_set_kpca[, 3], y_pred)

## Visualizing the training set results
library(ElemStatLearn)
set = training_set_kpca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = .01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = .01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM with KPCA (Training Set)',
     xlab = 'V1', ylab = 'V2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))

## Visualizing the test set results
library(ElemStatLearn)
set = test_set_kpca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = .01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = .01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM with KPCA (Test Set)',
     xlab = 'V1', ylab = 'V2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))
