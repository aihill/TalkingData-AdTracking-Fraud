#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:00:29 2018

@author: kazuki.onodera
"""

import numpy as np
import os
import gc
import lightgbm as lgb
from time import sleep
import utils
utils.start(__file__)

SEED = np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999

np.random.seed(SEED)
print('seed :', SEED)

# =============================================================================
# wait
# =============================================================================
while True:
    if os.path.isfile('SUCCESS_802'):
        break
    else:
        sleep(60*1)

utils.send_line('START {}'.format(__file__))

# =============================================================================
# load train
# =============================================================================

dtrain = lgb.Dataset('../data/dtrain.mt')
gc.collect()

# =============================================================================
# xgboost
# =============================================================================

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.1,
         'max_depth': 4,
         'num_leaves': 2**4-1,
         'colsample_bytree': 0.8,
         'subsample': 0.1,
         'nthread': 64,
         'seed': SEED}

gc.collect()

cv = lgb.cv(param, dtrain, NROUND, nfold=5, early_stopping_rounds=50, verbose_eval=10)



#==============================================================================
utils.end(__file__)

