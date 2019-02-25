#Machine learning using Regression
#Read the data

##>Data set 1: Rocks vs. Mines
## Independent variables: sonar soundings at different frequencies
## Dependent variable (target): Rock or Mine


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy
import random
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl
import pylab
from pandas import DataFrame
url="https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
df = pd.read_csv(url,header=None)
df.describe()

df[60]=np.where(df[60]=='R',0,1)

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.3)
x_train = train.iloc[0:,0:60]
y_train = train[60]
x_test = test.iloc[0:,0:60]
y_test = test[60]
y_train

model = linear_model.LinearRegression()
model.fit(x_train,y_train)

training_predictions = model.predict(x_train)
print(np.mean((training_predictions - y_train) ** 2))

print('Train R-Square:',model.score(x_train,y_train))
print('Test R-Square:',model.score(x_test,y_test))

print(max(training_predictions),min(training_predictions),np.mean(training_predictions))

def confusion_matrix(predicted, actual, threshold):
    if len(predicted) != len(actual): return -1
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for i in range(len(actual)):
        if actual[i] > 0.5: #labels that are 1.0  (positive examples)
            if predicted[i] > threshold:
                tp += 1.0 #correctly predicted positive
            else:
                fn += 1.0 #incorrectly predicted negative
        else:              #labels that are 0.0 (negative examples)
            if predicted[i] < threshold:
                tn += 1.0 #correctly predicted negative
            else:
                fp += 1.0 #incorrectly predicted positive
    rtn = [tp, fn, fp, tn]

    return rtn

testing_predictions = model.predict(x_test)

testing_predictions = model.predict(x_test)
confusion_matrix(testing_predictions,np.array(y_test),0.5)

cm = confusion_matrix(testing_predictions,np.array(y_test),0.5)
misclassification_rate = (cm[1] + cm[2])/len(y_test)
misclassification_rate

[tp, fn, fp, tn] = confusion_matrix(testing_predictions,np.array(y_test),0.5)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f_score = 2 * (precision * recall)/(precision + recall)
print('precision:', precision, 'recall:', recall, 'f_score:', f_score)

[tp, fn, fp, tn] = confusion_matrix(testing_predictions,np.array(y_test),0.9)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f_score = 2 * (precision * recall)/(precision + recall)
print('precision:', precision, 'recall:', recall, 'f_score:', f_score)


positives = list()
negatives = list()
actual = np.array(y_train)
for i in range(len(y_train)):

    if actual[i]:
        positives.append(training_predictions[i])
    else:
        negatives.append(training_predictions[i])
df_p = pd.DataFrame(positives)
df_n = pd.DataFrame(negatives)
fig, ax = plt.subplots()
a_heights, a_bins = np.histogram(df_p)
b_heights, b_bins = np.histogram(df_n, bins=a_bins)
width = (a_bins[1] - a_bins[0])/3
ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')
ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen')

positives = list()
negatives = list()
actual = np.array(y_test)
for i in range(len(y_test)):

    if actual[i]:
        positives.append(testing_predictions[i])
    else:
        negatives.append(testing_predictions[i])
df_p = pd.DataFrame(positives)
df_n = pd.DataFrame(negatives)
fig, ax = plt.subplots()
a_heights, a_bins = np.histogram(df_p)
b_heights, b_bins = np.histogram(df_n, bins=a_bins)
width = (a_bins[1] - a_bins[0])/3
ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')
ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen')

from sklearn.metrics import roc_curve, auc

(fpr, tpr, thresholds) = roc_curve(y_train,training_predictions)
area = auc(fpr,tpr)
pl.clf() #Clear the current figure
pl.plot(fpr,tpr,label="In-Sample ROC Curve with area = %1.2f"%area)

pl.plot([0, 1], [0, 1], 'k') #This plots the random (equal probability line)
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('In sample ROC rocks versus mines')
pl.legend(loc="lower right")
pl.show()
