#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:08:00 2020

@author: root
"""
import matplotlib.pyplot as plt 
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
    J = []
    while i <= iterations: 
        
        h = hypothesis(theta, X)

        term_update = ((h-y)*X).sum(axis=0)      
        theta = theta - (alpha *term_update/m) 
        J.append(cost(theta,m,X,y))
        i = i + 1 
        
    plt.plot(J,label = 'cost')
    plt.ylabel('cost', fontsize = 14)
    plt.xlabel('iteration', fontsize = 14)
    plt.show()
    print(J[-1])
        
    return theta