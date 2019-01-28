import numpy as np
import scipy.stats as stats

# Calculations of:
print("Taking the mean:" ,
      np.mean([1,2,3,4,5,6,7]))
print("Finding Standard Deviation:",
      np.std([1,2,3,4,5,6,7]))
print("Finding Square-root:",
      np.sqrt(1738))

alpha = .95
interval_end = 1-((1-alpha)/2)
print(interval_end)
#z_mult = stats.norm.ppf(interval_end)

x_bar = 1150
n = 22
alpha = .90
sd = 40

interval_end = 1-((1-alpha)/2)
z_mult = stats.norm.ppf(interval_end)

lower = x_bar - z_mult*(sd/np.sqrt(n))
upper = x_bar + z_mult*(sd/np.sqrt(n))

print(lower)
print(upper)
