## Upper Confidence Bracket

## Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

## Implementing the UCB
# Step 1: At each round n, we consider two numbers for each ad i
N = 10000
d = 10
ads_selected = integer(0)
numbers_of_selection = integer(d)   # << Ni(n)
sums_of_the_reward = integer(d)     # << Ri(n)
total_reward = 0

# Step 2: From these two numbers we compute:
for (n in 1:N) {
  max_upper_bound = 0
  ad = 0
  
  for (i in 1:d) {
    if (numbers_of_selection[i] > 0) {
      # - The average reward of ad i up to round n
      average_reward = sums_of_the_reward[i] / numbers_of_selection[i]
      # - The confidence interval at round n
      delta_i = sqrt(3 / 2 * log(n) / numbers_of_selection[i])
      
      upper_bound = average_reward + delta_i
    } else {
      upper_bound = 1e400
    }
    
    # Step 3: We select the ad i that has the maximum UCB
    if (upper_bound > max_upper_bound) {
      max_upper_bound = upper_bound
      ad = i
    }
  }
  
  ads_selected = append(ads_selected, ad)
  numbers_of_selection[ad] = numbers_of_selection[ad] + 1
  reward = dataset[n, ad]
  sums_of_the_reward[ad] = sums_of_the_reward[ad] + reward
  total_reward = total_reward + reward
}

## Visualizing the results
hist(
  ads_selected, col = 'blue',
  main = 'Histogram of Ads Selection', 
  xlab = 'Ads', ylab = 'Number of times each ad was selected'
)
