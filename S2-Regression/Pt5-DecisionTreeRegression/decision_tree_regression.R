## Importing the data set
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

## Fitting Decision Tree Regression into the data set
# install.packages('rpart')
library('rpart')
regressor = rpart(formula = Salary ~ ., 
                  data = dataset, control = 
                    rpart.control(minsplit = 1))

## Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

## Visualizing the Decision Tree Regression results
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
# install.packages('ggplot2')
library('ggplot2')
ggplot() +
  geom_point(aes(dataset$Level, dataset$Salary), colour = 'red') +
  geom_line(
    aes(x_grid, predict(regressor, newdata = data.frame(Level = x_grid))), 
    colour = 'blue'
  ) +
  ggtitle('Truth of Bluff (Decision Tree Regression)') +
  xlab('Position Level') +
  ylab('Salary')
