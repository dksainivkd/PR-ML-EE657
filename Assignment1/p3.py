import numpy as np
import pandas as pd
import math as m
import matplotlib.pyplot as plt

a=pd.read_csv('Wage_dataset.csv',sep=',', header=None)
tr_data=a.values

n=len(tr_data)
year=np.array(tr_data[:,0])
age=np.array(tr_data[:,1])
education=np.array(tr_data[:,4])
wage=np.array(tr_data[:,10])


# 1. Age vs wage  polynomial regression ********************************

print("1.Age vs Wage polynomial regression --->")
print("Enter the degree of polynomial for regression")
k=int(input())
k=k+1

A=np.zeros((k,k))
y=np.zeros(k)

for i in range(0,k):
    y[i]=np.sum(wage*age**i)
    for j in range(0,k):
        A[i][j]=np.sum(age**(i+j))      
W_age=np.matmul(np.linalg.inv(A),np.transpose(y))

r_wage=[]
for i in range(0,n):
    sm=0
    for j in range(0,k):
        sm=sm+W_age[j]*age[i]**j
    r_wage.append(sm)

plt.title('Wage vs Age ')
plt.ylabel('Wage')
plt.xlabel('Age')
plt.plot(age,wage,'bo',age,r_wage,'r^')
plt.show()

print("")
# 2. Year vs wage polynomial regression *********************************************

print("2.Year vs Wage polynomial regression --->")
print("Enter the degree of polynomial for regression")
k=int(input())
k=k+1

A=np.zeros((k,k))
y=np.zeros(k)
for i in range(0,k):
    y[i]=np.sum(wage*year**i)
    for j in range(0,k):
        A[i][j]=np.sum(year**(i+j))       
W_year=np.matmul(np.linalg.inv(A),np.transpose(y))

r_wage=[]
for i in range(0,n):
    sm=0
    for j in range(0,k):
        sm=sm+W_year[j]*year[i]**j
    r_wage.append(sm)

plt.title('Wage vs Year ')
plt.ylabel('Wage')
plt.xlabel('Year')
plt.plot(year,wage,'bo',year,r_wage,'r^')
plt.show()

print("")
# 3. Education vs wage regression *********************************

print("3.Education vs Wage polynomial regression --->")
print("Enter the degree of polynomial for regression ")
k=int(input())
k=k+1

A=np.zeros((k,k))
y=np.zeros(k)

for i in range(0,k):
    y[i]=np.sum(wage*education**i)
    for j in range(0,k):
        A[i][j]=np.sum(education**(i+j))       
W_ed=np.matmul(np.linalg.inv(A),np.transpose(y))

r_wage=[]
for i in range(0,n):
    sm=0
    for j in range(0,k):
        sm=sm+W_ed[j]*education[i]**j
    r_wage.append(sm)

plt.title('Wage vs Education ')
plt.ylabel('Wage')
plt.xlabel('Education')
plt.plot(education,wage,'bo',education,r_wage,'r^')
plt.show()




