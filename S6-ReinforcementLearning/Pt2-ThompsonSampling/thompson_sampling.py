## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

## Implementing the Thompson Sampling
# Step 1: At each round n, we consider two numbers for each ad i
import random
N = 100
d = 10
ads_selected = []
# - The number of times the ad i got reward 1 up to round n
numbers_of_rewards_1 = [0] * d    # << N^1i(n)
# - The number of times the ad i got reward 0 up to round n
numbers_of_rewards_0 = [0] * d    # << N^0i(n)
total_reward = 0

for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        ## Step 2: For each ad i, we take a random draw from the distribution
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        
        ## Step 3: Select the ad that has the highest random beta
        if (random_beta > max_random):
            max_random = random_beta
            ad = i
    
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1: 
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward += reward
        

## Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of Ads Selection')
plt.xlabel('Ads')
plt.ylabel('Number times each ad was selected')
plt.show()
