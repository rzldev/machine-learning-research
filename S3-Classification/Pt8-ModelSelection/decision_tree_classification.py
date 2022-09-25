## Importing the libraries
import numpy as np
import pandas as pd

## Importing the dataset
'''
Breast Cancer
Dependet Variables: Class
Independent Variables: Clump Thickness, Uniformity of Cell Size, Uniformity of Cell Shape,
    Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin, 
    Normal Nucleoli, Mitoses
'''
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

## Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Fitting the Decision Tree Classification model on the training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

## Predicting the test set
y_pred = classifier.predict(X_test)
# print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), 1))

## Making Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
