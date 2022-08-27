## Importing the data set
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

## Fitting Support Vector Regression into the data set
# install.packages('e1071')
library('e1071')
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression')

## Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

## Visualizing the Support Vector Regression results
# install.packages('ggplot2')
library('ggplot2')
ggplot() +
  geom_point(aes(dataset$Level, dataset$Salary), colour = 'red') +
  geom_line(aes(dataset$Level, predict(regressor, newdata = dataset)), colour = 'blue') +
  ggtitle('Truth or Bluff (Support Vector Regression)') +
  xlab('Position Level') +
  ylab('Salary')