#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 18:08:51 2018

@author: kazuki.onodera
"""

import pandas as pd
import numpy as np
from os import system
import os
from datetime import datetime
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
import lightgbm as lgb
import gc
from tqdm import tqdm
from multiprocessing import Pool
from glob import glob
import utils
utils.start(__file__)

SEED = 71 #np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999

np.random.seed(SEED)
print('seed :', SEED)

system('rm SUCCESS_802')

# =============================================================================
# load
# =============================================================================
#dtrain = lgb.Dataset('../data/dtrain.mt')
#gc.collect()
#
#
#dvalid = lgb.Dataset('../data/dvalid.mt')
#gc.collect()

# =============================================================================
# load
# =============================================================================
categorical_feature = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']

print('concat train')
load_files = sorted(glob('../data/*_train_sampling.p'))
X = pd.concat([pd.read_pickle(f) for f in tqdm(load_files)], axis=1)
print('X.isnull().sum().sum():', X.isnull().sum().sum())
drop_feature = ['click_time', 'attributed_time']
X.drop(drop_feature, axis=1, inplace=True)
X.fillna(-1, inplace=True)

print('train.shape:', X.shape )


y = X.is_attributed
del X['is_attributed']; gc.collect()

dtrain = lgb.Dataset(X, label=y, categorical_feature=categorical_feature)
X_head = X.head()

del X; gc.collect()





print('concat valid')
load_files = sorted(glob('../data/*_valid_sampling.p'))
X = pd.concat([pd.read_pickle(f) for f in tqdm(load_files)], axis=1)
print('X.isnull().sum().sum():', X.isnull().sum().sum())
drop_feature = ['click_time', 'attributed_time']
X.drop(drop_feature, axis=1, inplace=True)
X.fillna(-1, inplace=True)

print('valid.shape:', X.shape )


y = X.is_attributed
del X['is_attributed']; gc.collect()

dvalid = lgb.Dataset(X[X_head.columns], label=y, categorical_feature=categorical_feature)

del X; gc.collect()


# =============================================================================
# lgbm 1
# =============================================================================

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.05,
         'max_depth': 4,
         'num_leaves': 2**4-1,
         'max_bin': 100,
         'min_child_samples': 300,
         'min_child_weight': 0,
         'colsample_bytree': 0.8,
         'subsample': 0.1,
         'nthread': 64,
         'scale_pos_weight': 500,
         
         'seed': SEED
         }

gc.collect()

evals_result = {}

model = lgb.train(params=param, train_set=dtrain, num_boost_round=NROUND, 
                  valid_sets=[dtrain, dvalid], 
                  valid_names=['train','valid'], 
                  early_stopping_rounds=50, 
                  evals_result=evals_result, 
                  verbose_eval=10,
                  categorical_feature=categorical_feature
                  )


# =============================================================================
# lgbm 2
# =============================================================================

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.05,
         'max_depth': 4,
         'num_leaves': 2**4-1,
         'max_bin': 100,
         'min_child_samples': 300,
         'min_child_weight': 0,
         'colsample_bytree': 0.8,
         'subsample': 0.1,
         'nthread': 64,
         'scale_pos_weight': 100,
         
         'seed': SEED
         }

gc.collect()

evals_result = {}

model = lgb.train(params=param, train_set=dtrain, num_boost_round=NROUND, 
                  valid_sets=[dtrain, dvalid], 
                  valid_names=['train','valid'], 
                  early_stopping_rounds=50, 
                  evals_result=evals_result, 
                  verbose_eval=10,
                  categorical_feature=categorical_feature
                  )




system('touch SUCCESS_802')



#==============================================================================
utils.end(__file__)

