import numpy as np
import scipy.stats as stats


### Calculate the 95% confidence interval (with a t-distribution)
### of the data stored in the"observations" variable below
### Assign the lower bound as a number to "lower" and the upper bound as a number to "upper"

observations = [104, 148, 109, 104, 108, 120, 134, 129, 140, 128, 142, 113, 125, 111, 132, 133, 109, 107]
alpha = .95
n = len(observations) # find "n"
x_bar = np.mean(observations) # find "x_bar"- the sample mean
sd = np.std(observations) # find the sample standard deviation
t_mult = stats.t.interval (alpha, df = n-1)[1] # Find multiplier
lower = x_bar - t_mult * (sd / np.sqrt(n))
upper = x_bar + t_mult * (sd / np.sqrt(n))


