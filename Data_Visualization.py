import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

acc_df =  pd.read_csv('part_acc.csv')

# Look at first row:
acc_df.head(1)

acc_df.describe(include = "object").T

# Find unique values in term
print(".unique(); Notice the leading spaces;\n", acc_df['term'].unique(), sep = '')
print("\n.value_counts()\n", acc_df['term'].value_counts(), sep = "")
### Notice the leading spaces.

acc_df.loc[acc_df['term'] == ' 36 months', 'loan_amnt'].describe().T

### What is the mean `loan_amnt` for loans of 'grade' == 'A'
ans1 = acc_df[acc_df['grade'] == 'A']['loan_amnt'].mean()
print('The mean loan amount: $', ans1)

### What is the mean loan amount for loans with a grade of "C"
### using the group by function
ans2 = acc_df.groupby('grade')['loan_amnt'].mean()['C']
print('The mean loan amount for loans with a C grade rating: $', ans2)

### What is the mean loan amount for loans with a term of 36 months and a grade of "D"?
ans3 = acc_df.groupby(['term','grade'])['loan_amnt'].mean()[(' 36 months', 'D')]
print('Them mean loam amount with terms of 36 months: $', ans3)

### HOW MANY (count) loans in this dataset are of grade "A" where home_ownership is "OWN"
ans4 = acc_df.groupby(['grade', 'home_ownership'])['loan_amnt'].count()[("A", "OWN")]
print('Number of loans with grade A and home owner: ', ans4)

### What is the mean interest rate for loans where the `loan_amnt` is:
### Greater than or equal to 15,000, AND less than or equal to 17,000
def loan_amnt_groups(index):

    if acc_df.loc[index,'loan_amnt'] >17000:
        return "hi"
    elif acc_df.loc[index, 'loan_amnt'] <15000:
        return "low"
    else:
        return "med"
ans5 = acc_df.groupby(loan_amnt_groups)['int_rate'].mean()['med']

print('The mean interest rate for loans where the loan amount is >= 15,000 and <= 17,000: $', ans5)
