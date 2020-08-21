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
from sklearn.preprocessing import StandardScaler
import Metrics
# from sklearn.linear_model import SGDRegressor
import BatchGradientDescent
import Plotting
    
def getting_data(data_file):
    
    df_data =  pd.read_csv(data_file, sep=",", header=None)
    if df_data.shape[1] == 2:
    
        X = df_data.iloc[:, 0:-1].values.reshape(-1, 1)  # values converts it into a numpy array
        y = df_data.iloc[:, -1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    else:
        X = df_data.iloc[:, 0:-1].values  # values converts it into a numpy array
        y = df_data.iloc[:, -1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
        
    return X,y 

def split_dataSet(X,y):
    # split data
    X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=0)
    return X_train, X_test, y_train, y_test
   
    
def train_linearRegression(X_train, y_train):   

    # Create linear regression object
    #linear_regressor = SGDRegressor()
    linear_regressor = LinearRegression()  
    # Train the model using the training sets
    reg = linear_regressor.fit(X_train, y_train)
    return reg 

# getting data : 
data_file = "ex1data1.txt"
X,y = getting_data(data_file)


# plotting learning curve to visualize if the model has an high bias
train_sizes = [1, 10, 15, 20, 30,40, 50,60,70, 77]
Plotting.plot_learning_curve(X,y,train_sizes)


# PART 1 : linear regression by using sklearn.linear_model
# Splitting the data
X_train, X_test, y_train, y_test = split_dataSet(X,y)
# Train the model using the training sets

reg = train_linearRegression(X_train, y_train)
# Make predictions using the testing set
y_pred = reg.predict(X_test)
# Some metrics to evaluate the model
print("# 1) Method : linear regression by using sklearn.linear_model")
Metrics.metrics(y_test, y_pred)

Plotting.plot_linear_regression(X_test,y_test,y_pred)

# PART 2 a): linear regression with Batch Gradient Descent without splitting the data

# initialize fitting parameters
initial_theta = np.zeros(2)  
X = np.insert(X,0,values=1, axis=1)
#X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype))) #another way to insert a column into a numpy array
m = len(X)

# Applying Gradient Descent
iterations = 1500;
alpha = 0.01;
theta = BatchGradientDescent.gradientDescent(initial_theta,alpha,X,y,m, iterations)
#x_test = np.array([[1, 3.5],[1, 7]])
# Make predictions 
y_pred = BatchGradientDescent.hypothesis(theta, X)
# Some metrics to evaluate the model
print("# 2. a) Method :linear regression with Batch Gradient Descent without splitting the data")
Metrics.metrics(y, y_pred)



# PART 2 b): linear regression with Batch Gradient Descent by splitting the data
initial_theta = np.zeros(2) # initialize fitting parameters
X_train = np.insert(X_train,0,values=1, axis=1)
X_test = np.insert(X_test,0,values=1, axis=1)


# Applying Gradient Descent
iterations = 1500;
alpha = 0.01;

theta = BatchGradientDescent.gradientDescent(initial_theta,alpha,X_train,y_train,m, iterations)
# Make predictions using the testing set
y_pred = BatchGradientDescent.hypothesis(theta, X_test)

# Some metrics to evaluate the model
print("# 2. b) Method :linear regression with Batch Gradient Descent by splitting the data")
Metrics.metrics(y_test, y_pred)


# PART 3 : Linear regression with multiple variables

# getting data : 
data_file = "ex1data2.txt"
X,y = getting_data(data_file)


# plotting learning curve to visualize if the model has an high bias
train_sizes = [1, 5, 15, 20, 37]
Plotting.plot_learning_curve(X,y,train_sizes)


# 3. a) Not Splitting the data
# Feature Normalization
X_stand =  StandardScaler().fit_transform(X)
# Train the model using the training sets
reg = train_linearRegression(X_stand, y)
# Make predictions using the testing set
y_pred = reg.predict(X_stand)
# Some metrics to evaluate the model
print("# 3. a) Method : linear regression with multiple variables by using sklearn.linear_model - without splitting the data")
Metrics.metrics(y, y_pred)

#  3. b) Splitting the data
X_train_stand, X_test_stand , y_train, y_test = split_dataSet(X_stand,y)
# Train the model using the training sets
reg = train_linearRegression(X_train_stand, y_train)
# Make predictions using the testing set
y_pred = reg.predict(X_test_stand)
# Some metrics to evaluate the model
print("# 3. b) Method : linear regression with multiple variables by using sklearn.linear_model - splitting the data")
Metrics.metrics(y_test, y_pred)



#  3. c)Applying Gradient Descent
initial_theta = np.zeros(3) # initialize fitting parameters
X_stand = np.insert(X_stand,0,values=1, axis=1)
m = len(X_stand)
iterations = 1500;
alpha = 0.01;

theta = BatchGradientDescent.gradientDescent(initial_theta,alpha,X_stand,y,m, iterations)
# Make predictions using the testing set
y_pred = BatchGradientDescent.hypothesis(theta, X_stand)

# Some metrics to evaluate the model
print("# 3. c) Method :linear regression with multiple variables with Batch Gradient Descent without splitting the data")
Metrics.metrics(y, y_pred)



















