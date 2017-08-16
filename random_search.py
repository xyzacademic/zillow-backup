# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:32:41 2017

@author: xueyunzhe
"""

#==============================================================================
# Import library
#==============================================================================
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import scipy as sp
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.model_selection import RandomizedSearchCV

#==============================================================================
# Main
#==============================================================================

x_train = np.load('train_data.npy')
y_train = np.load('train_label.npy')
x_test = np.load('test_data.npy')
y_test= np.load('test_label.npy')

y_train = np.log(y_train)
y_train = np.exp(y_train)
y_mean = y_train.mean()

params = {
            'max_depth' : np.arange(2,10), 
            'learning_rate':np.linspace(0.01,0.1, 100), 
            'n_estimators':np.arange(30,150),
            'min_child_weight':[0.8,0.9, 1, 1.1, 1.2, 3], 
#            max_delta_step=0, 
            'subsample':[0.6,0.7,0.8,0.9,1], 
            'colsample_bytree':[0.8,0.9,1], 
#            colsample_bylevel=1, 
            'reg_alpha':np.linspace(0,1,1000), 
            'reg_lambda':np.linspace(0,1,1000), 
#            scale_pos_weight=1,
}


 
regressor = xgb.XGBRegressor(silent=True, objective='reg:linear', booster='gbtree',
                         n_jobs=1, gamma=0,  base_score=y_mean, random_state=0, 
                          missing=-1)

model= RandomizedSearchCV(estimator=regressor,
                          param_distributions=params,
                          n_iter=100,
                          scoring='neg_mean_absolute_error',
                          n_jobs=-1,
                          cv=5,
                          random_state=4396,
                          verbose=1
                          
                          )                         
                         
a= time.time()   
print('Start searching...')                      
model.fit(x_train, y_train)
print('Searching finished, cost %.0f seconds' %(time.time()-a))
k = model.predict(x_test)
eps= pd.Series(y_train).median() - pd.Series(model.predict(x_train)).median()
k = np.log(k+eps)
#k += eps

print(abs(y_test - k).mean())