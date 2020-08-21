#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:08:00 2020

@author: root
"""

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