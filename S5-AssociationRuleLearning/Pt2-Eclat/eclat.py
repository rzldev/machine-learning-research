## Importing the libraries
import numpy as np
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

## Data preprocessing
transactions = []
for i in range (0, len(dataset)): 
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

## Training the Eclat model on the dataset
from apyori import apriori
eclat = apriori(transactions=transactions, min_support=.003, min_confidence=.2, 
                min_lift=3, min_length=2, max_length=2)

## Visualizing 
# Displaying the first results coming directly from the output of the eclat function
results = list(eclat)
print(results)

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
    return list(zip(lhs, rhs, supports))

# Displaying the results non sorted
resultsInDataFrame = pd.DataFrame(data=inspect(results), columns=['Product 1', 'Product 2', 'Support'])
print(resultsInDataFrame)

# Displaying the results sorted by descending lifts
print(resultsInDataFrame.nlargest(n=10, columns='Support'))

