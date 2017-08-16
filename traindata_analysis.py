# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:08:22 2017

@author: X
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


df = pd.read_csv('train_data.csv')
df['label'] =df['logerror'].map(lambda x: 0 if x<0 else 1)
df['label2'] = abs(df.logerror.values)
df = df.loc[df.logerror>-0.4]
df = df.loc[df.logerror<0.4]
df_nan = df.fillna(-1)
feature_list = [ 'transaction_month', 'transaction_day',
       'airconditioningtypeid', 'architecturalstyletypeid', 'basementsqft',
       'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid',
       'buildingqualitytypeid', 'calculatedbathnbr', 'decktypeid',
       'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
       'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
       'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt',
       'fullbathcnt', 'garagecarcnt', 'garagetotalsqft', 'hashottuborspa',
       'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet',
       'poolcnt', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7',
       'propertycountylandusecode', 'propertylandusetypeid',
       'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity',
       'regionidcounty', 'regionidneighborhood', 'regionidzip', 'roomcnt',
       'storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid',
       'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'yearbuilt',
       'numberofstories', 'fireplaceflag', 'structuretaxvaluedollarcnt',
       'taxvaluedollarcnt', 'assessmentyear', 'landtaxvaluedollarcnt',
       'taxamount', 'taxdelinquencyflag', 'taxdelinquencyyear',
       'censustractandblock']
df_nan[feature_list] = df_nan[feature_list].applymap(lambda x: 1 if  x==-1 else 0)

p= df_nan[feature_list].astype('str').values.tolist()
for i in range(len(p)):
    p[i] = ''.join(p[i])
df_nan['uniques'] = p
df['uniques'] = p
k = pd.DataFrame({'uniques':df_nan['uniques'].value_counts().index.tolist(),
                  'counts':df_nan['uniques'].value_counts().tolist() })
k['cluster'] = k.index

df = df.merge(k[['uniques','cluster']], on='uniques')

def score(i):
    properties = df.loc[df.cluster==i]
    properties.dropna(axis=1, inplace=True)
    y = properties.logerror
    drop_list = ['ParcelId', 'logerror', 'transaction_month', 'transaction_day',
                 'label',
           'label2', 'uniques', 'cluster']
    properties.drop(drop_list, axis=1, inplace=True)
    for c in properties.columns:
        
        if properties[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(properties[c].values))
            properties[c] = lbl.transform(list(properties[c].values))
    x = properties.values
    y_mean = np.mean(y)
    
    xgb_params = {
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 1,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'base_score': y_mean,
        'silent': 1
    }
    dtrain = xgb.DMatrix(x, y)
    
    # cross-validation
    cv_result = xgb.cv(xgb_params, 
                       dtrain, 
                       nfold=5,
                       num_boost_round=200,
                       early_stopping_rounds=5,
    #                   verbose_eval=10, 
                       show_stdv=False
                      )
    print('I:  %d'%i)
    print(cv_result.iloc[-1][0])
    
for i in range(1):
    score(i)

#np.save('data.npy', x)
#np.save('label.npy',y)