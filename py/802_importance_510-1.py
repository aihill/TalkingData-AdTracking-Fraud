#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 12:02:36 2018

@author: kazuki.onodera
"""

import pandas as pd
import numpy as np
from os import system
import os
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
import lightgbm as lgb
import gc
from itertools import combinations
from time import sleep
import utils
utils.start(__file__)

TARGET_SEED = 3833

SEED = 71 #np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999

np.random.seed(SEED)
print('seed :', SEED)

system('rm SUCCESS_802')

categorical_feature = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']
categorical_feature += ['nearestNext_ip', 'nearestPre_ip', 'nearestNext_app', 
                        'nearestPre_app', 'nearestNext_device', 'nearestPre_device', 
                        'nearestNext_os', 'nearestPre_os', 'nearestNext_channel',
                        'nearestPre_channel']

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.05,
         'max_depth': 4,
         'num_leaves': 2**4-1,
         'max_bin': 100,
         'min_child_samples': 300,
         'min_child_weight': 100,
         'colsample_bytree': 0.1,
         'subsample': 0.1,
         'nthread': 64,
         'scale_pos_weight': 100,
         'bagging_freq': 1,
#         'lambda_l1': 3,
#         'lambda_l2': 3,
         
         'seed': SEED
         }

# =============================================================================
# wait
# =============================================================================
while True:
    if os.path.isfile('SUCCESS_801'):
        break
    else:
        sleep(60*1)

utils.send_line('START {}'.format(__file__))


# =============================================================================
# lgb
# =============================================================================
dtrain = lgb.Dataset(pd.read_feather(f'../data/X_train_mini_s{TARGET_SEED}.f'),
                     label=pd.read_feather(f'../data/y_train_mini_s{TARGET_SEED}.f').is_attributed,
                     categorical_feature=categorical_feature)

gc.collect()

dvalid = lgb.Dataset(pd.read_feather(f'../data/X_valid_mini_s{TARGET_SEED}.f'),
                     label=pd.read_feather(f'../data/y_valid_mini_s{TARGET_SEED}.f').is_attributed,
                     categorical_feature=categorical_feature)

gc.collect()

models = []
for i in range(3):
    evals_result = {}
    param.update({'seed': np.random.randint(9999)})
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

imp.to_csv(f'imp_{__file__}.csv', index=False)



#==============================================================================
system('touch SUCCESS_802')
utils.end(__file__)


