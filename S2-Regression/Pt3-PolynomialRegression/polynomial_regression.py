## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values


## Training the Linear Regression Model on the whole dataset
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x, y)


## Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_reg = PolynomialFeatures(degree=4)
x_poly = polynomial_reg.fit_transform(x)
linear_reg_2 = LinearRegression()
linear_reg_2.fit(x_poly, y)


## Visualizing the Linear Regression results
plt.scatter(x, y, color='red')
plt.plot(x, linear_reg.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


## Visualizing the Polynomial Regression results
plt.scatter(x, y, color='red')
plt.plot(x, linear_reg_2.predict(x_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


## Visualizing the Polynomial Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, linear_reg_2.predict(polynomial_reg.fit_transform(x_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


## Predicting a new result with Linear Regression
pred_salary = linear_reg.predict([[6], [6.5], [7]])
print(pred_salary)


## Predicting a new result with Polynomial Regression
pred_salary_2 = linear_reg_2.predict(polynomial_reg.fit_transform([[6], [6.5], [7]]))
print(pred_salary_2)
