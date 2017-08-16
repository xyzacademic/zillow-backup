# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 03:19:27 2017

@author: X
"""
#==============================================================================
# Import library
#==============================================================================
import numpy as np
import pandas as pd







#==============================================================================
# main
#==============================================================================
pf = pd.read_csv('properties_2016.csv')
pf.rename(columns={'parcelid':'ParcelId'}, inplace=True)
dtrain= pd.read_csv('train_2016_v2.csv', parse_dates=["transactiondate"])
dtrain.rename(columns={'parcelid':'ParcelId'}, inplace=True)
dtrain.drop(labels=['transactiondate'], axis=1, inplace=True)
dtrain= dtrain.merge(pf, on='ParcelId')

feature_list = pf.columns[1:]

dtrain.to_csv('train_data.csv', index=False)
pf.fillna(-1, inplace=True)
pf.to_csv('properties.csv', index=False)

pf_ = pf.applymap(lambda x: 0 if  x==-1 else 1)
pf_['ParcelId'] = pf['ParcelId']
pf_.to_csv('pf_.csv', index=False)
