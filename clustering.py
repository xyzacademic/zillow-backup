# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 03:35:52 2017

@author: X
"""
#==============================================================================
# Import library
#==============================================================================
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt


#==============================================================================
# Main
#==============================================================================
pf = pd.read_csv('pf_.csv')
dt = pd.read_csv('train_data.csv')
x = pf.values[:, 1:]


k = 20

km = KMeans(n_clusters=k, init='k-means++', n_init=10,
             max_iter=300, tol=0.0001, precompute_distances='auto', 
             verbose=0, random_state=4396, copy_x=False, n_jobs=1, 
             algorithm='auto')
print(km)

print('Start to fit...')
a = time.time()
km.fit(x)
b = time.time()
print('Clustering finished...')
print('Cost %d seconds' %np.around(b-a))

label_df = pd.DataFrame({
        'ParcelId': pf.ParcelId.values,
        'cluster':km.labels_,
        })

df = dt.merge(label_df, on='ParcelId')

if df.cluster.unique().shape[0] < label_df.cluster.unique().shape[0]:
    print('train_data\'s cluster number: %d' %df.cluster.unique().shape[0])
    print('Properties\'s cluster number: %d' %label_df.cluster.unique().shape[0])
    print('gather cluster...')
    
    a = time.time()
    centers = km.cluster_centers_
    temp = label_df.cluster.value_counts()
    p = df.cluster.value_counts()
    extension_label1 = set(p.loc[p.values<100].index)
    m = set(label_df.cluster.unique())
    n = set(df.cluster.unique())
    extension_label2  = m-n
    extension_label = extension_label2 | extension_label1
    
    def l1_distance(a,b):
        return abs(a-b).sum()
        
    replace_dict = {}
    for name in extension_label:
        min_index =0
        min_distance=25555
        for i in range(centers.shape[0]):
            if i not in extension_label:
                if l1_distance(centers[i],centers[name]) < min_distance:
                    min_distance = l1_distance(centers[i],centers[name])
                    min_index = i
        replace_dict[str(name)]= min_index
        
    label = label_df.cluster.values
    for i in range(label.shape[0]):
        if label[i] in extension_label:
            label[i] =  replace_dict[str(label[i])]
            
    label_df['cluster'] = label
    b = time.time()
    print('Gathering is finished...')
    pq = dt.merge(label_df, on='ParcelId')
    print('train_data\'s cluster number: %d' %pq.cluster.unique().shape[0])
    print('Properties\'s cluster number: %d' %label_df.cluster.unique().shape[0])
    print('Cost %d seconds' %np.around(b-a))

label_df.to_csv('cluster_label.csv', index=False)
print('Writing finished...')


#properties = properties.merge(label_df, on='ParcelId')#

plt.plot(np.arange(label_df.cluster.unique().shape[0]), label_df.cluster.value_counts().values)
plt.show()