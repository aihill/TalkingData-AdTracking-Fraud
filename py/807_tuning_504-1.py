#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 17:09:04 2018

@author: Kazuki
"""


import pandas as pd
import numpy as np
#from os import system
#import os
#from datetime import datetime
import sys
sys.path.append('/home/kazuki_onodera/Python')
#import lgbmextension as ex
#from GA import GA
import lightgbm as lgb
import gc
#from tqdm import tqdm
#from glob import glob
import utils
utils.start(__file__)
# =============================================================================

SEED = np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999
DO_SAMPLING = True
FRAC = 0.2


# =============================================================================
np.random.seed(SEED)
print('seed :', SEED)

#system('rm ../data/*sampling.f')
#system('rm SUCCESS_801')

train_files = range(66)
valid_files = range(78, 99)
print(f'train_files: {train_files}')
print(f'valid_files: {valid_files}')

categorical_feature = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']

# =============================================================================
# load
# =============================================================================
if DO_SAMPLING:
    X_train = pd.concat([ pd.read_pickle(f'../data/dtrain/{j:03d}.p').sample(frac=FRAC, random_state=SEED) for j in train_files ], 
                         ignore_index=True)
    y_train = pd.concat([ pd.read_pickle(f'../data/is_attributed/{j:03d}.p').sample(frac=FRAC, random_state=SEED) for j in train_files ], 
                         ignore_index=True)
    
    X_valid = pd.concat([ pd.read_pickle(f'../data/dtrain/{j:03d}.p') for j in valid_files ], 
                         ignore_index=True)
    y_valid = pd.concat([ pd.read_pickle(f'../data/is_attributed/{j:03d}.p') for j in valid_files ], 
                         ignore_index=True)

else:
    X_train = pd.concat([ pd.read_pickle(f'../data/dtrain/{j:03d}.p') for j in train_files ], 
                         ignore_index=True)
    y_train = pd.concat([ pd.read_pickle(f'../data/is_attributed/{j:03d}.p') for j in train_files ], 
                         ignore_index=True)
    
    X_valid = pd.concat([ pd.read_pickle(f'../data/dtrain/{j:03d}.p') for j in valid_files ], 
                         ignore_index=True)
    y_valid = pd.concat([ pd.read_pickle(f'../data/is_attributed/{j:03d}.p') for j in valid_files ], 
                         ignore_index=True)


categorical_feature_ = list( set(categorical_feature) & set(X_train.columns) )

dtrain = lgb.Dataset(X_train, label=y_train,
                     categorical_feature=categorical_feature_)

gc.collect()

dvalid = lgb.Dataset(X_valid, label=y_valid,
                     categorical_feature=categorical_feature_)

categorical_feature = list( set(categorical_feature) & set(X_train.columns) )
sample_size = X_train.shape[0]

print(f' sample_size: {sample_size}    categorical_feature:{categorical_feature}')

del X_train, y_train, X_valid, y_valid; gc.collect()

# =============================================================================
# lgb
# =============================================================================
def do_lgb(param):
    print(f'\n')
    print(param)
    evals_result = {}
    model = lgb.train(params=param, train_set=dtrain, num_boost_round=NROUND, 
                      valid_sets=[dtrain, dvalid], 
                      valid_names=['train', 'valid'], 
                      early_stopping_rounds=50, 
                      evals_result=evals_result, 
                      verbose_eval=False,
                      categorical_feature=categorical_feature
                      )
    
    train_score = model.best_score['train']['auc']
    valid_score = model.best_score['valid']['auc']
    
    print(f'train auc:{train_score}  valid auc:{valid_score}')
    
    return valid_score
# =============================================================================
# main
# =============================================================================
#parameters = [
#        {'min':2, 'max':5, 'type':int}, # max_depth
#        {'min':1, 'max':10000, 'type':int}, # scale_pos_weight
#        {'min':0.01, 'max':sample_size/1000, 'type':float, 'round':2}, # min_child_weight
#        {'min':0.4, 'max':1, 'type':float, 'round':2}, # subsample
#        {'min':0.4, 'max':1, 'type':float, 'round':2}, # colsample_bytree
#        {'min':0, 'max':10, 'type':float, 'round':1}, # lambda_l1
#        {'min':0, 'max':10, 'type':float, 'round':2}, # lambda_l2
#        ]

params = [
        {
         # fixed
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.2,
         'max_bin': 100,
         'nthread': 64,
         'bagging_freq': 10,
         
         # optimize
         'max_depth':        3,
         'num_leaves':       2**3-1,
         'scale_pos_weight': 100,
         'min_child_weight': 1,
         'subsample':        0.1,
         'colsample_bytree': 0.5,
         'lambda_l1':        0,
         'lambda_l2':        0,
         
         # fixed?
         'min_child_samples': 300,
         'seed': np.random.randint(9999)
         },
        
        {
         # fixed
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.2,
         'max_bin': 100,
         'nthread': 64,
         'bagging_freq': 10,
         
         # optimize
         'max_depth':        3,
         'num_leaves':       2**3-1,
         'scale_pos_weight': 300,
         'min_child_weight': 1,
         'subsample':        0.1,
         'colsample_bytree': 0.5,
         'lambda_l1':        0,
         'lambda_l2':        0,
         
         # fixed?
         'min_child_samples': 300,
         'seed': np.random.randint(9999)
         },
        
        {
         # fixed
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.2,
         'max_bin': 100,
         'nthread': 64,
         'bagging_freq': 10,
         
         # optimize
         'max_depth':        3,
         'num_leaves':       2**3-1,
         'scale_pos_weight': 500,
         'min_child_weight': 1,
         'subsample':        0.1,
         'colsample_bytree': 0.5,
         'lambda_l1':        0,
         'lambda_l2':        0,
         
         # fixed?
         'min_child_samples': 300,
         'seed': np.random.randint(9999)
         },
        
        {
         # fixed
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.2,
         'max_bin': 100,
         'nthread': 64,
         'bagging_freq': 10,
         
         # optimize
         'max_depth':        3,
         'num_leaves':       2**3-1,
         'scale_pos_weight': 100,
         'min_child_weight': 100,
         'subsample':        0.1,
         'colsample_bytree': 0.5,
         'lambda_l1':        0,
         'lambda_l2':        0,
         
         # fixed?
         'min_child_samples': 300,
         'seed': np.random.randint(9999)
         },
        
        {
         # fixed
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.2,
         'max_bin': 100,
         'nthread': 64,
         'bagging_freq': 10,
         
         # optimize
         'max_depth':        3,
         'num_leaves':       2**3-1,
         'scale_pos_weight': 100,
         'min_child_weight': 300,
         'subsample':        0.1,
         'colsample_bytree': 0.5,
         'lambda_l1':        0,
         'lambda_l2':        0,
         
         # fixed?
         'min_child_samples': 300,
         'seed': np.random.randint(9999)
         },
        
        {
         # fixed
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.2,
         'max_bin': 100,
         'nthread': 64,
         'bagging_freq': 10,
         
         # optimize
         'max_depth':        3,
         'num_leaves':       2**3-1,
         'scale_pos_weight': 100,
         'min_child_weight': 0.001,
         'subsample':        0.1,
         'colsample_bytree': 0.5,
         'lambda_l1':        5,
         'lambda_l2':        0,
         
         # fixed?
         'min_child_samples': 300,
         'seed': np.random.randint(9999)
         },
        
        {
         # fixed
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.2,
         'max_bin': 100,
         'nthread': 64,
         'bagging_freq': 10,
         
         # optimize
         'max_depth':        3,
         'num_leaves':       2**3-1,
         'scale_pos_weight': 100,
         'min_child_weight': 0.001,
         'subsample':        0.1,
         'colsample_bytree': 0.5,
         'lambda_l1':        0,
         'lambda_l2':        5,
         
         # fixed?
         'min_child_samples': 300,
         'seed': np.random.randint(9999)
         },
        
        
        ]

for param in params:
    do_lgb(param)






#==============================================================================
utils.end(__file__)

