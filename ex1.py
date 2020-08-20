#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 22:13:21 2020

@author: root
"""

# Machine Learning Online Class - Exercise 1: Linear Regression

# useful libraries
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
import Metrics
from sklearn.linear_model import SGDRegressor
import BatchGradientDescent
import Plotting
    

        
# getting data : 
data_file = "ex1data1.txt"
df_data =  pd.read_csv(data_file, sep=",", header=None)
df_data.columns = ["Population", "Profit"]
X = df_data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
y = df_data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column


# plotting learning curve to visualize if the model has an high bias
train_sizes = [1, 10, 15, 20, 30,40, 50,60,70, 77]
Plotting.plot_learning_curve(X,y,train_sizes)


# PART 1 : linear regression by using sklearn.linear_model
# split data
X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=0)
# Create linear regression object
#linear_regressor = SGDRegressor()
linear_regressor = LinearRegression()  
# Train the model using the training sets
reg = linear_regressor.fit(X_train, y_train)  
# Make predictions using the testing set
y_pred = linear_regressor.predict(X_test)  

# Some metrics to evaluate the model
print("# 1) Method : linear regression by using sklearn.linear_model")
Metrics.metrics(y_test, y_pred)

Plotting.plot_linear_regression(X_test,y_test,y_pred)

# PART 2 : linear regression with Batch Gradient Descent 

# compute and display initial cost
initial_theta = np.zeros(2) # initialize fitting parameters
X = np.insert(X,0,values=1, axis=1)
#X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype))) #another way to insert a column into a numpy array
m = len(X)


iterations = 1500;
alpha = 0.01;

theta = BatchGradientDescent.gradientDescent(initial_theta,alpha,X,y,m, iterations)
#x_test = np.array([[1, 3.5],[1, 7]])
y_pred = BatchGradientDescent.hypothesis(theta, X)
print("# 2) Method :linear regression with Batch Gradient Descent without splitting the data")
Metrics.metrics(y, y_pred)




initial_theta = np.zeros(2) # initialize fitting parameters
X_train = np.insert(X_train,0,values=1, axis=1)
X_test = np.insert(X_test,0,values=1, axis=1)



iterations = 1500;
alpha = 0.01;

theta = BatchGradientDescent.gradientDescent(initial_theta,alpha,X_train,y_train,m, iterations)

y_pred = BatchGradientDescent.hypothesis(theta, X_test)

# Some metrics to evaluate the model
print("# 3) Method :linear regression with Batch Gradient Descent by splitting the data")
Metrics.metrics(y_test, y_pred)