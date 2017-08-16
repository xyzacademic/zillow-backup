# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 21:37:15 2017

@author: xueyunzhe
"""

#
# This script is inspired by this discussion:
# https://www.kaggle.com/c/zillow-prize-1/discussion/33710
#

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import scipy as sp

#x = np.load('data.npy')
#y = np.load('label.npy')
#y_mean = np.mean(y)
#
#x = (x-x.mean(0))/x.std(0)
# xgboost params

d1 = np.load('d1.npy')
d2 = np.load('d2.npy')
d3 = np.load('d3.npy')
y = np.load('label.npy')
y_mean = np.mean(y)
x = np.concatenate((d1, d3), axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 1,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}
dtrain = xgb.DMatrix(x,y)
# cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   nfold=5,
                   num_boost_round=500,
                   early_stopping_rounds=5,
                   verbose_eval=10, 
                   show_stdv=False
                  )
#print(cv_result[:-1])
num_boost_rounds = len(cv_result)
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
k = model.predict(dtrain)
print(sp.stats.pearsonr(y,k)[0])
print(abs(y- k).mean())