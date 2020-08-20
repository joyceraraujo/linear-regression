#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:13:52 2020

@author: root
"""
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
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