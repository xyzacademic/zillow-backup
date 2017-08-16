# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:47:16 2017

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
import os 


#==============================================================================
# Function define
#==============================================================================

def predict(df, model, eps):
    feature = df.drop(['ParcelId', 'cluster'], axis=1)
    x = feature.values
    k = model.predict(x)
    k = k-eps
#    k = np.log(k+eps)
    
    return pd.DataFrame({
                         'ParcelId':df.ParcelId.values,
                         'logerror':k
                         })


def load(cluster_index, model_folder):
    fname = 'cluster_%d_model.xgb'%cluster_index
    save_path = os.path.join(model_folder, fname)    
    with open(save_path,'rb') as f:
        temp = pickle.load(f)
    
    model = temp['model']
    eps = temp['eps']
    return model, eps





#==============================================================================
# Main
#==============================================================================
properties = pd.read_csv('properties.csv')
label_df = pd.read_csv('cluster_label.csv')

number_cluster = label_df.cluster.unique().shape[0]
for c in properties.columns:    
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(properties[c].values.tolist())
        properties[c] = lbl.transform(properties[c].values.tolist())
        
properties = properties.merge(label_df, on='ParcelId')

concat_list = []
model_folder = 'checkpoint%d'%number_cluster

for i in properties.cluster.unique():
    df = properties.loc[properties.cluster==i]
    model, eps= load(i, model_folder)
#    eps=0
    concat_list.append(predict(df, model, eps))
    print('cluster %d prediction is finished...' %i)
    
dt = pd.concat(concat_list, axis=0, ignore_index=True)
output = pd.DataFrame({'ParcelId': dt['ParcelId'].astype(np.int32),
        '201610': dt.logerror.values, '201611': dt.logerror.values, '201612': dt.logerror.values,
        '201710': dt.logerror.values, '201711': dt.logerror.values, '201712': dt.logerror.values})
        
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]

output.to_csv('submission_%d'%number_cluster, index=False)