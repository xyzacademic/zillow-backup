#
# This script is inspired by this discussion:
# https://www.kaggle.com/c/zillow-prize-1/discussion/33710
#

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
#    y_train = np.exp(y_train)
    
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
                              n_iter=100,
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
    
    
    
properties = pd.read_csv('properties_2016.csv')
train = pd.read_csv("train_2016_v2.csv")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1).values
x_test = properties.drop(['parcelid'], axis=1).values
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.4 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1).values
y_train = train_df["logerror"].values.astype(np.float32)
y_train = np.exp(y_train)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

model, eps = model_select(x_train, y_train)

k = model.predict(x_test)
k =  np.log(k)
output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': k, '201611': k, '201612': k,
        '201710': k, '201711': k, '201712': k})
        
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]

output.to_csv('submission_sstt', index=False)