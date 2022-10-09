## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

## Data Preprocessing
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])
   
## Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions=transactions, min_support=.003, min_confidence=.2, min_lift=3,
                min_lenth=2, max_length=2)

## Visualizing the results

# Displaying the first results coming directly from the output of the apriori function
results = list(rules)
#print(results)

# Putting the result well organized into a Pandas Dataframe
"""
RelationRecord(
    items=frozenset({'chicken', 'light cream'}), 
    support=0.004532728969470737, 
    ordered_statistics=[OrderedStatistic(
        items_base=frozenset({'light cream'}), 
        items_add=frozenset({'chicken'}), 
        confidence=0.29059829059829057, 
        lift=4.84395061728395
    )]
)
"""
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

## Displaying the results non sorted
resultsInDataFrame = pd.DataFrame(data=inspect(results), columns=[
    'Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
print(resultsInDataFrame)

## Displaying the results sorted by descending lifts
print(resultsInDataFrame.nlargest(n=10, columns='Lift'))
