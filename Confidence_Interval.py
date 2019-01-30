import numpy as np
import scipy.stats as stats

### GRADED
### Calculate the 95% confidence interval (with a Z- (NORMAL) DISTRIBUTION)
### of the difference of the means of the collections stored in obs1 and obs2
### NOTE: Specifically find the CI for the mean of obs1 - mean of obs2
### Assign the lower bound as a number to "lower" and the upper bound as a number to "upper"
### Answers will be tested to three decimal places
### YOUR ANSWER BELOW
obs1 = [22.9 , 26.08, 25.04, 22.09, 24.28, 31.3 , 25.47, 24.17, 23.42,
25.64, 23.96, 23.94, 25.35, 20.92, 27.74, 25.93, 26.9 , 27.87,
22.43, 23.73, 29.25, 25.66, 23.6 , 26.77, 17.38, 26.26, 17.67,
24.04, 19.42, 27.41, 30.02, 25.22, 26.47, 24.47, 22.85, 20.07,
29.46, 23.61, 26.54, 25.37]
obs2 = [26.37, 32.62, 22.13, 22.64, 32.33, 25.62, 18.69, 26.86, 17.87,
18.16, 26.37, 25.77, 22.57, 27.41, 17.2 , 22.61, 26.97, 28.78,
24.02, 25.41, 27.88, 28.99, 30.06, 30.23, 24.19, 17.06, 24.38,
24.13, 25.87, 31.58, 21.19, 32.07, 30.07, 24.23, 27.37]
alpha = .95
# Calculate sample means
mean_x = np.mean(obs1)
mean_y = np.mean(obs2)
# Calculate sample standard deviations
sd_x = np.std(obs1)
sd_y = np.std(obs2)
# Count number of observations in each sample
n_x = len(obs1)
n_y = len(obs2)
# Calculate Observed Difference of means
diff = mean_x - mean_y
# Calculate Standard Error
se = np.sqrt( (sd_x**2/n_x) + (sd_y**2/n_y))
# Find z-multiplier
z_mult = stats.norm.interval(alpha)[1]
# Calculate confidence interval
lower = diff - z_mult * se
upper = diff + z_mult * se

print('lower endpoint: ', lower)
print('upper endpoint: ', upper)
