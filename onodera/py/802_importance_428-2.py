#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:07:30 2018

@author: Kazuki
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
from glob import glob
import utils
utils.start(__file__)

SEED = 71 #np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999

np.random.seed(SEED)
print('seed :', SEED)

system('rm SUCCESS_802')

categorical_feature = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']

# =============================================================================
# load
# =============================================================================
dtrain = lgb.Dataset(pd.read_feather('../data/X_train.f'),
                     label=pd.read_feather('../data/y_train.f').values,
                     categorical_feature=categorical_feature)
gc.collect()


dvalid = lgb.Dataset(pd.read_feather('../data/X_valid.f'),
                     label=pd.read_feather('../data/y_train.f').values,
                     categorical_feature=categorical_feature)
gc.collect()

# =============================================================================
# load
# =============================================================================

#print('concat train')
#load_files = sorted(glob('../data/*_train_sampling.p'))
#X = pd.concat([pd.read_pickle(f) for f in tqdm(load_files)], axis=1)
#print('X.isnull().sum().sum():', X.isnull().sum().sum())
#drop_feature = ['click_time', 'attributed_time']
#X.drop(drop_feature, axis=1, inplace=True)
#X.fillna(-1, inplace=True)
#
#print('train.shape:', X.shape )
#
#
#y = X.is_attributed
#del X['is_attributed']; gc.collect()
#
#dtrain = lgb.Dataset(X, label=y, categorical_feature=categorical_feature)
#X_head = X.head()
#
#del X; gc.collect()
#
#
#
#
#
#print('concat valid')
#load_files = sorted(glob('../data/*_valid_sampling.p'))
#X = pd.concat([pd.read_pickle(f) for f in tqdm(load_files)], axis=1)
#print('X.isnull().sum().sum():', X.isnull().sum().sum())
#drop_feature = ['click_time', 'attributed_time']
#X.drop(drop_feature, axis=1, inplace=True)
#X.fillna(-1, inplace=True)
#
#print('valid.shape:', X.shape )
#
#
#y = X.is_attributed
#del X['is_attributed']; gc.collect()
#
#dvalid = lgb.Dataset(X[X_head.columns], label=y, categorical_feature=categorical_feature)
#
#del X; gc.collect()


# =============================================================================
# lgb
# =============================================================================

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.1,
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

models = []
for i in range(3):
    param.update({'seed':np.random.randint(9999)})
    model = lgb.train(params=param, train_set=dtrain, num_boost_round=NROUND, 
                      valid_sets=[dtrain, dvalid], 
                      valid_names=['train','valid'], 
                      early_stopping_rounds=50, 
                      evals_result=evals_result, 
                      verbose_eval=10,
                      categorical_feature=categorical_feature
                      )
    models.append(model)

imp = ex.getImp(models)

t = datetime.today()
date = t.date()
hour = t.hour
imp.to_csv(f'imp_{date}-{hour:02d}h.csv', index=False)


system('touch SUCCESS_802')



#==============================================================================
utils.end(__file__)

