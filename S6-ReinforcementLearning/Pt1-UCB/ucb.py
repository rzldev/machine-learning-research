## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


## Implementing UCB
# Step 1: At each round n, we consider two numbers for easch ad i
N = 10000
d = 10
ads_selected = []
numbers_of_selecting = [0] * d   # << Ni(n)
sums_of_rewards = [0] * d        # << Ri(n)
total_reward = 0

# Step 2: From these two numbers we compute:
import math
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    
    for i in range(0, d):
        if numbers_of_selecting[i] > 0:
            # - The average reward of ad i up to round n
            average_reward = sums_of_rewards[i] / numbers_of_selecting[i]   # << ri(n)
            # - The Confidence Interval at round n
            delta_i = math.sqrt(3/2 * math.log(n) / numbers_of_selecting[i])    # << Δi(n)
            
            # Step 3: We select the ad i that has the maximum UCB    
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i

    ads_selected.append(ad)
    numbers_of_selecting[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
            

## Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of Ads Selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
