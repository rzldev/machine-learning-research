## Thompson Sampling

## Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

## Implementing the Thompson Sampling
# Step 1: At each round n, consider two numbers for each ad i
N = 500
d = 10
ads_selected = integer(0)
# - The number of times the ad i got reward 1 up to round n
numbers_of_rewards_0 = integer(d)
# - The number of times the ad i got reward 0 up to round n
numbers_of_rewards_1 = integer(d)
total_reward = 0

for (n in 1:N) {
  max_random = 0
  ad = 0
  for (i in 1:d) {
    # Step 2: For each ad i, we take a random draw from the distribution
    random_beta = rbeta(n = 1, 
                        shape1 = numbers_of_rewards_1[i] + 1, 
                        shape2 = numbers_of_rewards_0[i] + 1)
    
    # Step 3: Select the ad that has the highest random beta
    if (random_beta > max_random) {
      max_random = random_beta
      ad = i
    }
  }
  
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  if (reward == 1) {
    numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
  } else {
    numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
  }
  total_reward = total_reward + reward
}

## Visualizing the results
hist(
  ads_selected, col = 'blue',
  main = 'Histogram of Ads Selection',
  xlab = 'Ads', ylab = 'Number of times each ad was selected'
)
