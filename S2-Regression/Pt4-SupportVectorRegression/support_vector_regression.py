## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## Importing the dataset
dataset = pd.read_csv('Position_salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


## Feature scaling
y = y.reshape(len(y), 1)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


## Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x, y)


## Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))
print(y_pred)


## Visualizing the SVR results
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1, 1)), color='blue')
plt.title('Truth or Bluff (Support Vector Regressor)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


## Visualizing the SVR results (for higher resolution and smoother curve)
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), .1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)).reshape(-1, 1)), color='blue')
plt.title('Truth or Bluff (Support Vector Regressor)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
