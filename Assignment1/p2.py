import pandas as pd
import numpy as np
import math as m
import matplotlib

a=pd.read_csv('P2_test.csv',sep=',',header=None)
ts_data=a.values
a=pd.read_csv('P2_train.csv',sep=',',header=None)
tr_data=a.values

row_train=len(tr_data)
col_train=len(tr_data[0])-1

# seperate data and label
ts_label=ts_data[:,2]
ts_data=ts_data[:,0:2]


no_c0=0
no_c1=0
X0=[]       # X0 belongs to class 0 and X1 belongs to class 1
X1=[]
for i in range(0,row_train):
    if(tr_data[i][2]==0):
        X0.append(tr_data[i][0:2])
        no_c0=no_c0+1
    else:
        X1.append(tr_data[i][0:2])
        no_c1=no_c1+1
X0=np.matrix(X0)
X1=np.matrix(X1)
        
mean0=np.matrix(np.mean(X0,axis=0))
mean1=np.matrix(np.mean(X1,axis=0))
cov0=np.matmul(np.transpose(X0-mean0),(X0-mean0))/(no_c0)
cov1=np.matmul(np.transpose(X1-mean1),(X1-mean1))/(no_c1)

"""***************make change in covarience **************** """
#print("****")
#print(cov0)
#print(cov1)
#print("****")
#cov0=np.eye(2)
#cov0[0][0]=3
#cov0[0][1]=4
#cov0[1][0]=2
#cov0[1][1]=7
#cov1=np.eye(2)
#cov1[0][0]=14
#cov1[0][1]=10
#cov1[1][0]=12
#cov1[1][1]=16


#******gaussian calculation

P_c0     = no_c0/row_train
P_c1     = no_c1/row_train              
cov0_inv = np.linalg.inv(cov0)
cov0_det = m.sqrt(abs(np.linalg.det(cov0)))
cov1_inv = np.linalg.inv(cov1)
cov1_det = m.sqrt(abs(np.linalg.det(cov1)))

pi_d     = (m.sqrt(2*m.pi))**col_train
row_test = len(ts_data)

conf_mat=[[0,0],[0,0]]

for i in range(0,row_test):
    x=ts_data[i]-mean0
    g=np.matmul(x,cov0_inv)
    h=np.matmul(g,np.transpose(x))
    PxC0=np.exp(-h/2)/(pi_d*cov0_det)
    
    x=ts_data[i]-mean1
    g=np.matmul(x,cov1_inv)
    h=np.matmul(g,np.transpose(x))
    PxC1=np.exp(-h/2)/(pi_d*cov1_det)

    PC0x=PxC0*P_c0/(PxC0*P_c0+PxC1*P_c1)
    PC1x=PxC1*P_c1/(PxC0*P_c0+PxC1*P_c1)
    
    if(PC0x>PC1x and ts_label[i]==0):
        conf_mat[0][0]=conf_mat[0][0]+1
    elif(PC0x>PC1x and ts_label[i]==1):
        conf_mat[0][1]=conf_mat[0][1]+1
    elif(PC0x<PC1x and ts_label[i]==0):
        conf_mat[1][0]=conf_mat[1][0]+1
    else:
        conf_mat[1][1]=conf_mat[1][1]+1

# confusion matrix
print("confusion matrix is : - ")
print(conf_mat)

ts_no_c0=0
ts_no_c1=0
for i in range(0,row_test):
    if(ts_label[i]==0):
        ts_no_c0=ts_no_c0+1
    else:
        ts_no_c1=ts_no_c1+1

 
conf_mat[0][0]=(conf_mat[0][0]/ts_no_c0)*100
conf_mat[0][1]=(conf_mat[0][1]/ts_no_c1)*100
conf_mat[1][0]=(conf_mat[1][0]/ts_no_c0)*100
conf_mat[1][1]=(conf_mat[1][1]/ts_no_c1)*100
print("percentage matrix ")
print(conf_mat)

"""-------------------------Ploting Contour---------------------------"""
#ploting contour
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


cov0=np.array(cov0)
cov1=np.array(cov1)

mean0=np.array(mean0)
mean1=np.array(mean1)

"""--------function for gaussian calculation------------"""
def bi_normal(position, mean, covar):
    n = mean.shape[0]
    covar_det = np.linalg.det(covar)
    covar_inv = np.linalg.inv(covar)
    d = np.sqrt((2*np.pi)**n * covar_det)
    fac = np.einsum('...k,kl,...l->...', position-mean, covar_inv, position-mean)
    return np.exp(-fac / 2) / d
"""-----------function end----------------------------"""
x_range=np.linspace(min(ts_data[:,0]),max(ts_data[:,0]),1000)
y_range=np.linspace(min(ts_data[:,1]),max(ts_data[:,1]),1000)
X, Y = np.meshgrid(x_range,y_range)

d = np.empty(X.shape + (2,))

d[:, :, 0] = X
d[:, :, 1] = Y
Z0 = bi_normal(d,mean0,cov0)
plt.contour(X,Y,Z0)
Z1 = bi_normal(d,mean1,cov1)
plt.contour(X,Y,Z1)


Z = (Z1 - Z0)
plt.contour(X,Y,Z)

plt.title('Iso-probability contour (case4 a1=3,b1=4,c1=2,d1=7/ a2=14,b2=10,c2=12,d2=16)')
plt.scatter(ts_data[:,0],ts_data[:,1])
plt.ylabel('x2')
plt.xlabel('x1')
plt.show()
