#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:12:30 2020

@author: root
"""
from sklearn.metrics import mean_squared_error, r2_score

def metrics(y_test, y_pred):
    
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
    # The mean squared error
    print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
    