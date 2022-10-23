## Artificial Neural Networks for Regression


## Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf


## Part 1 - Data preprocessing
# Importing the dataset
dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)


## Part 2 - Building the ANN
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the ouput layer
ann.add(tf.keras.layers.Dense(units=1))


## Part 3 - Training the ANN
# Compiling the ANN
ann.compile(optimizer='adam', loss='mean_squared_error')

# Training the ANN model on the training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Predicting the test set results
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), 1))
