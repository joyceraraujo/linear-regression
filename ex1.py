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
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.linear_model import SGDRegressor


# function to plot learning curve
def plot_learning_curve(X,y,train_sizes):
    
    train_sizes, train_scores, validation_scores = learning_curve(
    estimator = LinearRegression(),
    X = X,
    y = y, train_sizes = train_sizes, cv = 5,
    scoring = 'neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1 )
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('MSE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)
    plt.legend()
    plt.show() 
    
# function to plot the result of linear regression  
def plot_linear_regression(X_test,y_test,y_pred):
    
    plt.scatter(X_test, y_test,  color='red')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.ylabel('Profit', fontsize = 14)
    plt.xlabel('Population', fontsize = 14)
    plt.title('Linear regression model', fontsize = 18, y = 1.03)
    plt.show()
    
def metrics(y_test, y_pred):
    
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
    # The mean squared error
    print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
#  functions to use in Gradient Descent method
def hypothesis(theta, x):

    h = (theta*x).sum(axis=1).reshape(-1,1)
    
    return h

def cost(theta,m,X,y):
    h =  hypothesis(theta, X)
    J = (h-y)**2
    J = J.sum()/(2*m)
    return J

def gradientDescent(initial_theta,alpha,X,y,m, iterations):
    i = 1
    theta = initial_theta
    while i <= iterations: 
        h = hypothesis(theta, X)
        term_update = ((h-y)*X).sum(axis=0)      
        theta = theta - (alpha *term_update/m) 
        i = i + 1 
        
    return theta


        
# getting data : 
data_file = "ex1data1.txt"
df_data =  pd.read_csv(data_file, sep=",", header=None)
df_data.columns = ["Population", "Profit"]
X = df_data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
y = df_data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column


# plotting learning curve to visualize if the model has an high bias
train_sizes = [1, 10, 15, 20, 30,40, 50,60,70, 77]
plot_learning_curve(X,y,train_sizes)


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
metrics(y_test, y_pred)

plot_linear_regression(X_test,y_test,y_pred)

# PART 2 : linear regression with Batch Gradient Descent 

# compute and display initial cost
initial_theta = np.zeros(2) # initialize fitting parameters
X = np.insert(X,0,values=1, axis=1)
#X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype))) #another way to insert a column into a numpy array
m = len(X)


iterations = 1500;
alpha = 0.01;

theta = gradientDescent(initial_theta,alpha,X,y,m, iterations)
#x_test = np.array([[1, 3.5],[1, 7]])
y_pred = hypothesis(theta, X)
print("# 2) Method :linear regression with Batch Gradient Descent without splitting the data")
metrics(y, y_pred)




initial_theta = np.zeros(2) # initialize fitting parameters
X_train = np.insert(X_train,0,values=1, axis=1)
X_test = np.insert(X_test,0,values=1, axis=1)



iterations = 1500;
alpha = 0.01;

theta = gradientDescent(initial_theta,alpha,X_train,y_train,m, iterations)

y_pred = hypothesis(theta, X_test)

# Some metrics to evaluate the model
print("# 3) Method :linear regression with Batch Gradient Descent by splitting the data")
metrics(y_test, y_pred)