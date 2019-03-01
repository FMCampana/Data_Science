#Predicting wine quality using a decision tree
#Decision tree regressors are used when the target variable is continuous and ordered (wine quality from 0 to 10)

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
import pandas as pd
from pandas import DataFrame
w_df = pd.read_csv(url,header=0,sep=';')
w_df.describe()

from sklearn.model_selection import train_test_split
train, test = train_test_split(w_df, test_size = 0.3)
x_train = train.iloc[0:,0:11]
y_train = train[['quality']]
x_test = test.iloc[0:,0:11]
y_test = test[['quality']]

#Use all data for cross validation
x_data = w_df.iloc[0:,0:11]
y_data = w_df[['quality']]
#x_data
y_test

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

model = DecisionTreeRegressor(max_depth = 3)
model.fit(x_train,y_train)

#Get the R-Square for the predicted vs actuals on the text sample
print("Training R-Square",model.score(x_train,y_train))
print("Testing R-Square",model.score(x_test,y_test))

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#from sklearn.cross_validation import cross_val_score
#from sklearn.cross_validation import KFold
crossvalidation = KFold(n_splits=5,shuffle=True, random_state=1)

from sklearn import tree
import numpy as np
for depth in range(1,10):
    model = tree.DecisionTreeRegressor(
    max_depth=depth, random_state=0)
    if model.fit(x_data,y_data).tree_.max_depth < depth:
        break
    score = np.mean(cross_val_score(model, x_data, y_data,scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1))
    print ('Depth: %i Accuracy: %.3f' % (depth,score))
