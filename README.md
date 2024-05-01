# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the data file and import numpy, matplotlib and scipy.
2.Visulaize the data and define the sigmoid function, cost function and gradient descent.
3.Plot the decision boundary.
4.Calculate the y-prediction. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: HARSSHITHA LAKSHMANAN
RegisterNumber: 212223230075

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'Placement_Data.csv')
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset['gender'] = dataset['gender'].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1+np.exp(-z))
def loss(theta ,X ,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iters):
    m=len(y)
    for i in range(num_iters):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -=alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iters=1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
 
*/
```

## Output:
![logistic regression using gradient descent](sam.png)
![image](https://github.com/harshulaxman/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145686689/bc8bf92b-8d26-410b-805b-435ee8fd4a8c)
![image](https://github.com/harshulaxman/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145686689/a795d598-0a51-4d54-980b-bbbee9232496)
![image](https://github.com/harshulaxman/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145686689/7f8bc294-4998-4aad-9d80-39c6df43daae)
![image](https://github.com/harshulaxman/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145686689/4f392dda-67bb-491f-a01e-90b3576ed470)
![image](https://github.com/harshulaxman/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145686689/ab90598a-4126-455b-9c3b-b473c7d55608)
![image](https://github.com/harshulaxman/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145686689/187338a2-a96f-444e-9f06-4d22bf22c69a)
![image](https://github.com/harshulaxman/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145686689/8e5bef95-ea71-4461-a1d7-57dd3d8dcab8)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

