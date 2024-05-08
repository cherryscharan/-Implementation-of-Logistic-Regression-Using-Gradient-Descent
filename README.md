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
```
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
```

## Output:
### Read the file and display
![WhatsApp Image 2024-05-09 at 01 00 12_270dd9cc](https://github.com/cherryscharan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/a689db78-0883-45f0-a3ae-d6553695892f)


### Categorizing columns
![WhatsApp Image 2024-05-09 at 01 00 21_6a451183](https://github.com/cherryscharan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/2b496fd3-6be2-4090-a581-ba25af79197d)


### Labelling columns and displaying dataset
![WhatsApp Image 2024-05-09 at 01 00 27_77ec9d62](https://github.com/cherryscharan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/b72ee9c1-afb0-4f10-b953-cf6dc8c530b9)


### Display dependent variable
![WhatsApp Image 2024-05-09 at 01 00 30_f4071dad](https://github.com/cherryscharan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/4efb20cd-b6eb-40ea-bf9d-0b2ac5bd80e4)

### Printing accuracy
![WhatsApp Image 2024-05-09 at 01 00 35_aa3a3a43](https://github.com/cherryscharan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/e9395656-fff7-4ffa-9b1a-7686c5852c25)


### Printing Y
![WhatsApp Image 2024-05-09 at 01 00 38_a28ca475](https://github.com/cherryscharan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/8e266670-6977-44f5-9cc0-2cd0cfd43164)



### Printing y_prednew
![WhatsApp Image 2024-05-09 at 01 00 43_75c45d9c](https://github.com/cherryscharan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146930617/bc7773fc-3dd4-4c55-99dd-496b22fb3c98)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
