# #### In this problem i used numpy, pandas, sklearn and graphviz 

# ## Importing Libraries

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn import decomposition
import graphviz


# # Reading data from file
# I have read data by using pandas "read_csv" module
# where  trainX = Traing data
#        trainY = Training Label
#        testX  = Test data
#        testY  = Test Label

trainX=pd.read_csv('trainX.csv',sep=',',header=None)
trainY=pd.read_csv('trainY.csv',sep=',',header=None)
testX=pd.read_csv('testX.csv',sep=',',header=None)
testY=pd.read_csv('testY.csv',sep=',',header=None)

# ### By using gini and entropy criterion

# # 1. criterion = gini
# To classify i used DecisionTreeClassifier

clf_gini = DecisionTreeClassifier(criterion ='gini')
clf_gini = clf_gini.fit(trainX,trainY)
y_pred = clf_gini.predict(testX)

# ### Calculating confusion matrix and Accuracy
# confusion_matrix() is a function which return a confusion matrix

print("")
print("Criterion = gini")
c=confusion_matrix(testY, y_pred)
print("Confusion matrix is by using whole data (criterion=gini):- ")
print(c)
print("Accuracy :",accuracy_score(testY,y_pred)*100)
print("Number of nodes are by using whole data (criterion=gini) = ",clf_gini.tree_.node_count)
l = clf_gini.tree_.children_left==-1
ct=0
for k in l:
    if(k==True):
        ct+=1
print("Number of leaf nodes = ",ct)

# By using gini criterion Accuracy is approx 85.96%  
# ## Plotting tree model

dot_data = tree.export_graphviz(clf_gini, out_file=None, filled=True, rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("ByGini")


# # 2. criterion = entropy
# To classify i used DecisionTreeClassifier

clf_entropy = DecisionTreeClassifier(criterion ='entropy')
clf_entropy = clf_entropy.fit(trainX,trainY)
y_pred = clf_entropy.predict(testX)

# ### Calculating confusion matrix and Accuracy
# confusion_matrix() is a function which return a confusion matrix
print("")
print("Criterion = entropy")
c=confusion_matrix(testY, y_pred)
print("Confusion matrix is by using whole data (criterion=entropy):- ")
print(c)
print("Accuracy :",accuracy_score(testY,y_pred)*100)
print("Number of nodes are by using whole data (criterion=entropy) = ",clf_gini.tree_.node_count)
l = clf_entropy.tree_.children_left==-1
ct=0
for k in l:
    if(k==True):
        ct+=1
print("Number of leaf nodes = ",ct)

# We have found that by using entropy criterion accuracy is increased to approx 91.31%

# ## Plotting tree model

dot_data = tree.export_graphviz(clf_entropy, out_file=None,filled=True, rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("ByEntropy")


# # By applying PCA

# I applied PCA and reduce the dimension from 30 to lower dimension  here i used loop for differet-different dimensions

# # 1. criterion = gini

#print("Variation in accuracy by applying PCA ")
#print("criterion=gini")
for i in range(29,0,-1):
    # pca on traing set
    pca = decomposition.PCA(n_components=i)
    pca.fit(trainX)
    trainX_pca = pca.transform(trainX)
    
    # pca on test data
    pca = decomposition.PCA(n_components=i)
    pca.fit(testX)
    testX_pca = pca.transform(testX)
    
    clf_gini = DecisionTreeClassifier(criterion ='gini')
    clf_gini = clf_gini.fit(trainX_pca,trainY)
    y_pred = clf_gini.predict(testX_pca)
    c=confusion_matrix(testY, y_pred)
    #print("Dimension ",i, " , Accuracy : ",accuracy_score(testY,y_pred)*100)
    #print("")

#print("")

# # 2. criterion = entropy

#print("criterion=entropy")
for i in range(29,0,-1):
    # pca on traing set
    pca = decomposition.PCA(n_components=i)
    pca.fit(trainX)
    trainX_pca = pca.transform(trainX)
    
    # pca on test data
    pca = decomposition.PCA(n_components=i)
    pca.fit(testX)
    testX_pca = pca.transform(testX)
    
    clf_gini = DecisionTreeClassifier(criterion ='entropy')
    clf_gini = clf_gini.fit(trainX_pca,trainY)
    y_pred = clf_gini.predict(testX_pca)
    c=confusion_matrix(testY, y_pred)
    #print("Dimension ",i," , Accuracy :- ",accuracy_score(testY,y_pred)*100)
    #print("")
#print("")

# # Second part binary decision tree with increasing size of training set

# ### Plot to show how accuracies vary with number of training samples.

test_acc=[]
train_acc=[]
no_train_sample=[]
c=1.0
for i in range(1,11):
    c=c-0.1
    X_train, X_test, Y_train, Y_test = train_test_split(trainX, trainY, test_size = c, random_state = 100)
    clf_gini = DecisionTreeClassifier(criterion = "gini")
    clf_gini.fit(X_train,Y_train)
    y_pred_test = clf_gini.predict(testX)
    y_pred_train = clf_gini.predict(trainX)

    test_acc.append(accuracy_score(testY,y_pred_test)*100)
    train_acc.append(accuracy_score(trainY,y_pred_train)*100)
    no_train_sample.append(len(X_train))

import matplotlib.pyplot as plt
plt.xlabel("Number of Training sample ")
plt.ylabel("accuracy")
plt.title("accuracy vs number of training sample")
plt.plot(no_train_sample,test_acc,label='test')
plt.plot(no_train_sample,train_acc,label='train')
plt.legend()
plt.show()
