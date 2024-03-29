## XGBoost

## Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

## Encoding the categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Male', 'Female'),
                                   labels = c(1, 2)))

## Splitting the dataset into the training set and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = .8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

## Fitting the XGBoost model on the training set
#install.packages('xgboost')
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-11]), 
                     label = training_set$Exited,
                     nrounds = 10)

## Predicting the test set results
y_pred = predict(classifier, newdata = as.matrix(test_set[-11]))
y_pred = (y_pred >= 0.5)

## Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)
(cm[1, 1] + cm[2, 2]) / (cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2])

## Applying k-Fold Cross Validation
#install.packages('caret')
library(caret)
folds = createFolds(training_set$Exited, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x,]
  test_fold = test_set[x,]
  classifier = xgboost(data = as.matrix(training_fold[-11]), 
                       label = training_fold$Exited,
                       nrounds = 10)
  y_pred = predict(classifier, as.matrix(test_fold[-11]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 11], y_pred)
  accuracy = (cm[1, 1] + cm[2, 2]) / (cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
accuracy
