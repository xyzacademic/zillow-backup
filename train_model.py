# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:59:31 2017

@author: xueyunzhe
"""

#==============================================================================
# Import library
#==============================================================================
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import scipy as sp
import matplotlib.pyplot as plt
import os
import pickle
import time
import shutil

#==============================================================================
# Function define
#==============================================================================

def data(cluster_index):
    df = properties.loc[properties.cluster==cluster_index]
    y_train = df.logerror.values
    y_train = np.exp(y_train)
    
    feature = df.drop(['ParcelId', 'logerror', 'cluster'], axis=1)
    x_train  = feature.values
    print('Cluster %d'  %cluster_index)
    return x_train, y_train
    
def model_select(x_train, y_train):
    
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
                              n_iter=400,
                              scoring='neg_mean_absolute_error',
                              n_jobs=-1,
                              cv=5,
                              random_state=4396,
                              verbose=0
                              
                              )      
                              
    a= time.time()   
    print('Start searching...')                      
    model.fit(x_train, y_train)
    print('Searching finished, cost %.0f seconds' %(time.time()-a))
    eps = pd.Series(y_train).median() - pd.Series(model.predict(x_train)).median()
#    print('train_mean: %f'%abs(np.log(y_train)).mean())
#    print('Score: %f' %model.best_score_)
    return model, eps
    
def save_model(model, eps, cluster_index, model_folder):
    fname = 'cluster_%d_model.xgb'%cluster_index
    save_path = os.path.join(model_folder, fname)
    
    temp= {'model': model,
           'eps': eps,
           }
    with open(save_path, 'wb') as f:
        pickle.dump(temp, f)
    print('Save %s successful...'%fname)
    
    
#==============================================================================
# Main
#==============================================================================
model_folder = 'checkpoint'

label_df = pd.read_csv('cluster_label.csv')
properties = pd.read_csv('train_data.csv')
properties.fillna(-1, inplace=True)

for c in properties.columns:    
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(properties[c].values.tolist())
        properties[c] = lbl.transform(properties[c].values.tolist())
        
properties = properties.merge(label_df, on='ParcelId')
properties = properties.loc[abs(properties.logerror)<0.4]


if os.path.exists(model_folder):
    shutil.rmtree(model_folder)
    print('Delete the older folder')
    os.mkdir(model_folder)
    print('Create %s successful'%model_folder)
else:
    os.mkdir(model_folder)
    print('Create %s successful'%model_folder)
    
number = properties.cluster.unique().shape[0]    
m = time.time()
counter = 0
for i in properties.cluster.unique():
    x, y = data(i)
    model, eps = model_select(x, y)
    save_model(model, eps, i, model_folder)
    counter += 1
    if counter %10 == 0:
        print(counter)
    
print('All cluster training finished, cost %.0f seconds'%(time.time() - m))
shutil.copytree(model_folder, model_folder+str(number))