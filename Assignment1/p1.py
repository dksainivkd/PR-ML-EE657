import pandas as pd
import numpy as np
import math as m

a=pd.read_csv('P1_data_test.csv',sep=',',header=None)
ts_data=a.values
a=pd.read_csv('P1_labels_test.csv',sep=',',header=None)
ts_label=a.values

a=pd.read_csv('P1_data_train.csv',sep=',',header=None)
tr_data=a.values
a=pd.read_csv('P1_labels_train.csv',sep=',',header=None)
tr_label=a.values

row_train=len(tr_data)
col_train=len(tr_data[0])

no_c5=0
no_c6=0
X5=[]
X6=[]
for i in range(0,row_train):
    if(tr_label[i]==5):
        X5.append(tr_data[i])
        no_c5=no_c5+1
    else:
        X6.append(tr_data[i])
        no_c6=no_c6+1

mean5=np.mean(X5,axis=0)
mean6=np.mean(X6,axis=0)
cov5=np.matmul(np.transpose(X5-mean5),(X5-mean5))/(no_c5-1)
cov6=np.matmul(np.transpose(X6-mean6),(X6-mean6))/(no_c6-1)

#*******************
"""change in covarience """
#cov6=cov5

#*****************

P_c5=no_c5/row_train
P_c6=no_c6/row_train
               
cov5_inv=np.linalg.inv(cov5)
cov5_det=m.sqrt(abs(np.linalg.det(cov5)))

cov6_inv=np.linalg.inv(cov6)
cov6_det=m.sqrt(abs(np.linalg.det(cov6)))

pi_d=(m.sqrt(2*m.pi))**col_train
row_test=len(ts_data)

conf_mat=[[0,0],[0,0]]

for i in range(0,row_test):
    x=ts_data[i]-mean5
    g=np.matmul(x,cov5_inv)
    h=np.matmul(g,np.transpose(x))
    PxC5=np.exp(-h/2)/(pi_d*cov5_det)

    x=ts_data[i]-mean6
    g=np.matmul(x,cov6_inv)
    h=np.matmul(g,np.transpose(x))
    PxC6=np.exp(-h/2)/(pi_d*cov6_det)
    
    PC5x=PxC5*P_c5/(PxC5*P_c5+PxC6*P_c6)
    PC6x=PxC6*P_c6/(PxC5*P_c5+PxC6*P_c6)
    
    if(PC5x>PC6x and ts_label[i]==5):
        conf_mat[0][0]=conf_mat[0][0]+1
    elif(PC5x>PC6x and ts_label[i]==6):
        conf_mat[0][1]=conf_mat[0][1]+1
    elif(PC5x<PC6x and ts_label[i]==5):
        conf_mat[1][0]=conf_mat[1][0]+1
    else:
        conf_mat[1][1]=conf_mat[1][1]+1
        
print("")
print("2X2 confusion matrix for cov6=cov5 ")
print(np.matrix(conf_mat))

ts_no_c5=0
ts_no_c6=0
for i in range(0,row_test):
    if(ts_label[i]==5):
        ts_no_c5=ts_no_c5+1
    else:
        ts_no_c6=ts_no_c6+1


conf_mat[0][0]=(conf_mat[0][0]/ts_no_c5)*100
conf_mat[0][1]=(conf_mat[0][1]/ts_no_c6)*100
conf_mat[1][0]=(conf_mat[1][0]/ts_no_c5)*100
conf_mat[1][1]=(conf_mat[1][1]/ts_no_c6)*100
print("")
print("Percentage for case cov6=cov5 ")
print(np.matrix(conf_mat))
print("")


    
