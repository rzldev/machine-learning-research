## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
'''
Combined Cycle Power Plant
Dependent variable: Energy output
Independent Variable: Ambient Temparature, Exhaust Vaccum, Ambient Pressure, Relative Humility
'''
dataset = pd.read_csv('Combined_Cycle_Power_Plant.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

## Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

## Training the Polynomial Regression model on the training set
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
polynomial_reg = PolynomialFeatures(degree=4)
X_poly = polynomial_reg.fit_transform(X_train)
linear_reg = LinearRegression()
linear_reg.fit(X_poly, y_train)

## Predicting the tests results
y_pred = linear_reg.predict(polynomial_reg.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

## Evaluating the model performance
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
