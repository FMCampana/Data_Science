import numpy as np
import scipy.stats as stats

### Given point estimate (x_bar), confidence level(Z[alpha/2]), n, and sample standard deviation,
### calculate a confidence interval.
### Assign the lower bound as a number to "lower" and the upper bound as a number to "upper"
### Calculate a 90% confidence interval where the sample mean of 22 observations was 150 with a
### sample standard deviation of 40

x_bar = 150
n = 22
alpha = .90
sd = 40

interval_end = 1-((1-alpha)/2)
z_mult = stats.norm.ppf(interval_end)
lower = x_bar - z_mult*(sd/np.sqrt(n))
upper = x_bar + z_mult*(sd/np.sqrt(n))

print(upper)
print(lower)
