
# coding: utf-8

# # Description:-
# In this problem bikes data is given the goal is to predict the daily level of bicycle rentals from environmental and seasonal variables using decision trees. first i arrange the data to use and then split data into training set and test data. Build a model on training set and then tested the test data and found the mean square error, number of nodes in regression tree, number of leaf nodes and finally plot the regression tree. In second part recode the months acccording to problem and again build a model on training set, tested it on test data and compared both results.

# #### In this problem i used numpy, pandas, sklearn and graphviz libraries

# # Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from sklearn import tree
from sklearn import metrics 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# # Reading data from file
# 
# I have read data by using pandas "read_csv" module.
df=pd.read_csv('bikes.csv')




# ### Rearranging data (Delete date column)

feature_names=list(df.columns.values)
feature_names=feature_names[1:11]
data=df.values
data=np.delete(data,(0),axis=1)

# ### Split data into training set and test data (size of test data = 25% of whole data)
train_data,test_data=train_test_split(data,test_size=0.25)


# ## seperating of train_data into trainX and train
trainX=train_data[:,0:10]
trainY=train_data[:,10]
testX=test_data[:,0:10]
testY=test_data[:,10]
# # Building model on training set
# using DecisionTreeregressor
clf = DecisionTreeRegressor(random_state=0)
clf.fit(trainX,trainY)


# #### clf.feature_importances_  function  gives the importance of all features. Higher value means that feature have more priority

feature_weight=clf.feature_importances_

# making list of features with features weight
l1=[]
for i in range(0,10):
    l2=[]
    l2.append(feature_names[i])
    l2.append(feature_weight[i])
    l1.append(l2)

# sorting with respect to feature weight

# #### Important features in decreasing order

l1=sorted(l1,key=lambda x: x[1])
l1.reverse()
print("")
print("Important variable in Decreasing order of priority :-")
print(l1)
print("")

# ### number of nodes
n_nodes = clf.tree_.node_count
print("Number of nodes = ",n_nodes)

# ## Calculation for number of leaf nodes
leaf= clf.tree_.children_left == -1
#No. of leaf nodes
ct=0
for k in leaf:
    if(k==True):
        ct+=1
print("Number of leaf nodes = ",ct)

# #### Classification of test data by using regression tree and printing the MSE

y_pred = clf.predict(testX)
print('Mean Squared Error:', metrics.mean_squared_error(testY, y_pred))  

# ## Plotting of regression tree

dot_data = tree.export_graphviz(clf, out_file=None,feature_names=feature_names,filled=True, rounded=True,
                                special_characters=True)  
graph = graphviz.Source(dot_data)
print('Number of nodes are :-',clf.tree_.node_count)
graph.render('part1')


# ##### Into 12 different groups week days divide the data

# # Second part of problem
print("")
print("Second part")
# ## Recoding the months 
for i in range(0,len(trainX)):
    if(trainX[i,2]==1 or trainX[i,2]==2):
        trainX[i,2]=1
    elif(trainX[i,2]==3 or trainX[i,2]==4 or trainX[i,2]==11 or trainX[i,2]==12):
        trainX[i,2]=3
    else:
        trainX[i,2]=2

for i in range(0,len(testX)):
    if(testX[i,2]==1 or testX[i,2]==2):
        testX[i,2]=1
    elif(testX[i,2]==3 or testX[i,2]==4 or testX[i,2]==11 or testX[i,2]==12):
        testX[i,2]=3
    else:
        testX[i,2]=2

# # Building model on training set
# using DecisionTreeregressor
clf = DecisionTreeRegressor(random_state=0)
clf.fit(trainX,trainY)

# #### clf.feature_importances_  function  gives the importance of all features. Higher value means that feature have more priority

feature_weight=clf.feature_importances_

# #### Important features in decreasing order

l1=[]
for i in range(0,10):
    l2=[]
    l2.append(feature_names[i])
    l2.append(feature_weight[i])
    l1.append(l2)
    
l1=sorted(l1,key=lambda x: x[1])
l1.reverse()
print("Important variable in Decreasing order of priority :-")
print(l1)
print("")


# ### number of nodes
n_nodes = clf.tree_.node_count
print("Number of nodes = ",n_nodes)

# ## Calculation for number of leaf nodes

leaf = clf.tree_.children_left == -1
#No. of leaf nodes
ct=0
for k in leaf:
    if(k==True):
        ct+=1
print("Number of leaf nodes = ",ct)

# ### Classification of test data by using regression tree and printing the MSE

y_pred = clf.predict(testX)
print('Mean Squared Error:', metrics.mean_squared_error(testY, y_pred))  

# # Plotting of regression tree

dot_data = tree.export_graphviz(clf, out_file=None,feature_names=feature_names,filled=True, rounded=True,
                                special_characters=True)  
graph = graphviz.Source(dot_data)
print('Number of nodes are :-',clf.tree_.node_count)
graph.render('part2')

