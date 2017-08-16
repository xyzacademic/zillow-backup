# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

label_df = pd.read_csv('cluster_label.csv')
properties = pd.read_csv('train_data.csv')
properties.fillna(-1, inplace=True)

for c in properties.columns:    
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(properties[c].values.tolist())
        properties[c] = lbl.transform(properties[c].values.tolist())
        
properties = properties.merge(label_df, on='ParcelId')

df = properties.loc[properties.cluster==8]
df = df.loc[abs(df.logerror)<0.4]
split=1000
y_train = df.logerror.values[:-split]
y_train = np.exp(y_train)
feature = df.drop(['ParcelId', 'logerror', 'cluster'], axis=1)

x_train  = feature.values[:-split, :]
y_mean = y_train.mean()
weight = np.tanh(abs(y_train - y_mean))*1000
x_test = feature.values[-split:,:]
y_test = df.logerror.values[-split:]
xgb_params = {
    'eta': 0.02,
    'max_depth':6,
    'subsample': 0.8,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train, weight=None)
dtest = xgb.DMatrix(x_test)

# cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   nfold=5,
                   num_boost_round=200,
                   early_stopping_rounds=5,
                   verbose_eval=10, 
                   show_stdv=False
                  )
num_boost_rounds = len(cv_result)
print(cv_result)
print(num_boost_rounds)

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
k = model.predict(dtest)

print(sp.stats.pearsonr(y_test,k)[0])
pd.Series(y_test).plot(kind='kde')
eps= pd.Series(y_train).median() - pd.Series(model.predict(dtrain)).median()
#eps=0
k = np.log(k+eps)
#k = k+eps
pd.Series(k).plot(kind='kde')
plt.show()
#print(abs(np.log(y_train) - np.log(k+eps)).mean())

print(abs(y_test - k).mean())
xgb.XGBRegressor()