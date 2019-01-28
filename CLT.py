import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define file_path containing data with string
data_path = "../resource/asnlib/publicdata/office_supply.csv"
# Read in data with Pandas
office = pd.read_csv(data_path)
# Create some example data
test_list = [1,5,6,7,6,2,5,7,8,3,6]
test_tuple = tuple(test_list)
# Use a variety of the `numpy` functions.
print("np.mean of 'office': \n", np.mean(office),"\n" ,sep = "")
print("\nnp.min of 'office['sales']': ", np.min(office['sales']))
print("\nnp.max of 'office['transactions']': ", np.max(office['transactions']))
print("\nnp.var of test_list: ", np.var(test_list))
print("\nnp.ptp of test_tuple: ", np.ptp(test_tuple))
print("\nnp.sqrt of 72: ", np.sqrt(72))
