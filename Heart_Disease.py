#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:36:18 2020

@author: anthony
"""

'''
Things to solve:
    - Use all data
    - Identify dependent variable (binary?)
    - Identify relevant independent variables
    - Apply machine learning algorithms
    - Compare accuracy of ML algorithms
'''


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

column = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']


df = pd.DataFrame([],columns = column)

indices = [2,3,8,9,11,15,18,31,37,39,40,43,50,57]



with open("heart-disease.names") as col_names:
    colum = col_names.read()
    
print(colum)


# FOR HUNGARIAN, SWITZERLAND, L0NG-BEACH-VA
with open("hungarian.data") as myfile:
    data = myfile.readlines()
    #print(data)
    #print(len(data))
temp_list = []
temp_list2 = []


for i in range(len(data)):
    temp = data[i].split()
    #print(temp)
    if 'name' in temp:
        temp = temp[0:-1]
        for j in range(len(temp)):
            temp_list.append(float(temp[j]))
        for k in indices:
            temp_list2.append(temp_list[k])
        #print(temp_list)
        aseries = pd.Series(temp_list2, index = column) #, index = df.columns
        #print(aseries)
        df = df.append(aseries, ignore_index = True)
        #print(df)
        temp_list = []
        temp_list2 = []
    else:
        for j in range(len(temp)):
            temp_list.append(float(temp[j]))

'''
# FOR PROCESSED.CLEVELAND
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


# ASK LUDOVICO
# 1. convert into array or list?
# 2. handle negative/missing/? values
# 3. randomize order of instances
# 4. split into train and test
# 5. try modeling one dataset first, Hungarian
# 6. try logistic or naive bayes, first, then try different statistical models
# 7. evaluate accuracy
# 8. combine datasets into one? or create separate models? try both?
# 9. improve code last
        
# feature selection?
# column name abstraction?



X = df.loc[:, df.columns != 'num'].to_numpy()
y = df['num'].to_numpy()
#X_train, X_test, y_train, y_test = train_test_split(df , y, test_size = 0.3, random_state = 0)

#print(X_train.shape)
#print(X_test.shape)


'''
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

    