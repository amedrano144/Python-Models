#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anthony
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# initialize dataframe
column = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
df = pd.DataFrame([],columns = column)
indices = [2,3,8,9,11,15,18,31,37,39,40,43,50,57]


# for HUNGARIAN, SWITZERLAND, L0NG-BEACH-VA datasets
with open("hungarian.data") as myfile:
    data = myfile.readlines()
temp_list = []
temp_list2 = []

# clean and organize data
for i in range(len(data)):
    temp = data[i].split()
    if 'name' in temp:
        temp = temp[0:-1]
        for j in range(len(temp)):
            temp_list.append(float(temp[j]))
        for k in indices:
            temp_list2.append(temp_list[k])
        aseries = pd.Series(temp_list2, index = column) #, index = df.columns
        df = df.append(aseries, ignore_index = True)
        temp_list = []
        temp_list2 = []
    else:
        for j in range(len(temp)):
            temp_list.append(float(temp[j]))

# select features to keep/drop based on proportion of missing values
count_missing_values = [0]*df.shape[1]
for m in range(df.shape[0]):
    for n in range(df.shape[1]):
        if df.iloc[m,n] < 0:
            count_missing_values[n] = count_missing_values[n] + 1
keep_indices = [1]*df.shape[1]
for a, b in enumerate(count_missing_values):
    if b > df.shape[0]/2:
        keep_indices[a] = 0
        
        
# create new dataframe with select features     
filtered_df = pd.DataFrame()
for x in range(df.shape[1]):
    if keep_indices[x] == 1:
        name = column[x]
        filtered_df[name] = df.iloc[:,x]



'''
# for PROCESSED.CLEVELAND dataset
with open("processed.cleveland.data") as myfile:
    data = myfile.readlines()
    #print(data)
    #print(len(data))
temp_list = []


for i in range(len(data)):
    temp = data[i].replace("\n",'')
    temp = temp.replace("?",'-9.0')
    temp = temp.split(',')
    #print(temp)


    for j in range(len(temp)):
        temp_list.append(float(temp[j]))

    #print(temp_list)
    aseries = pd.Series(temp_list, index = column) #, index = df.columns
    #print(aseries)
    df = df.append(aseries, ignore_index = True)
    #print(df)
    temp_list = []
'''

# convert dataframe to array and split data
X = df.loc[:, df.columns != 'num'].to_numpy()
y = df['num'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.1)


# initialize classifier
#gnb = GaussianNB()
#gnb = LogisticRegression(max_iter = 20000)
gnb = SVC()

# train classifier
model = gnb.fit(X_train, y_train)

# make predictions
preds = gnb.predict(X_test)
print(preds)

# evaluate accuracy
print("accuracy:",accuracy_score(y_test, preds))





'''
# feature selection?
with open("heart-disease.names") as col_names:
    colum = col_names.read()
# column name abstraction?
plt.figure(figsize = (12,10))
cor = X_train.corr()
sns.heatmap(cor, annot = True, cmap = plt.cm.CMRmap_r)
plt.show()


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.7)
len(set(corr_features))
'''