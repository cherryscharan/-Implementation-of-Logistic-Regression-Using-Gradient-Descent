# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for sigmoid, loss, gradient and predict and perform operations. 

## Program:

/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Charan kumar S
RegisterNumber: 212223220015
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset
dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes



dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

Y

theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
    
y_pred = predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)


## Output:
### Read the file and display
![WhatsApp Image 2024-05-09 at 01 00 12_78e88efa](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/7b343950-80d2-4281-b154-98d6325769b4)


### Categorizing columns
![WhatsApp Image 2024-05-09 at 01 00 21_105915b0](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/e3dc20d2-8961-4e7a-9c0b-8d55dad556da)


### Labelling columns and displaying dataset
![WhatsApp Image 2024-05-09 at 01 00 27_c2243928](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/a0ca4105-b258-47cd-8f16-54783368c2ca)


### Display dependent variable
![WhatsApp Image 2024-05-09 at 01 00 30_3a8a3b7b](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/179146e5-c7e9-47c1-9816-19c5aea66979)

### Printing accuracy
![WhatsApp Image 2024-05-09 at 01 00 35_a281f98b](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/d343453a-a62c-46df-97d9-6cdb6bc4bc22)


### Printing Y
![WhatsApp Image 2024-05-09 at 01 00 38_10c89cf1](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/5a6dca7c-d220-4d3f-a42c-a7e4c4639dca)



### Printing y_prednew
![WhatsApp Image 2024-05-09 at 01 00 43_ea918e61](https://github.com/AkilaMohan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/59a55396-3f2a-4856-aa56-ec2b73bc47ac)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
