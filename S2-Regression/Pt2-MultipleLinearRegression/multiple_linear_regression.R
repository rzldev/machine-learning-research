## Importing the dataset
dataset = read.csv('50_Startups.csv')


## Encoding categorical data
dataset$State = factor(
  dataset$State, levels = c('New York', 'California', 'Florida'), labels = c(1, 2, 3)
)


## Splitting the dataset into training set and test set
# install.packages('caTools')
library('caTools')
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 3/4)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


## Fitting Multiple Linear Regression to the training set
regressor = lm(formula = Profit ~ ., data = training_set)
summary(regressor)


## Predicting the test set results
y_pred = predict(regressor, newdata = test_set)
y_pred


## Building the optimal model using Backward Elimination
regressor = lm(
  formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
  data = dataset
)
regressor = lm(
  formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
  data = dataset
)
regressor = lm(
  formula = Profit ~ R.D.Spend + Marketing.Spend,
  data = dataset
)
regressor = lm(
  formula = Profit ~ R.D.Spend,
  data = dataset
)
summary(regressor)