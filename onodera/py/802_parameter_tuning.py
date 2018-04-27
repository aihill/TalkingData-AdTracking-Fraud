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

dtrain = lgb.Dataset('../data/dtrain.mt')
gc.collect()


dvalid = lgb.Dataset('../data/dvalid.mt')
gc.collect()


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
                  verbose_eval=10
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
                  verbose_eval=10
                  )




system('touch SUCCESS_802')



#==============================================================================
utils.end(__file__)

