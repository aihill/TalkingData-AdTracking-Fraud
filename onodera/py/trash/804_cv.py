#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:00:29 2018

@author: kazuki.onodera
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys
sys.path.append('/home/kazuki_onodera/Python')
import xgbextension as ex
import os
#from multiprocessing import Process, Pipe
import gc
import xgboost as xgb
from time import sleep
import utils
utils.start(__file__)

SEED = 4308 # np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999

np.random.seed(SEED)
print('seed :', SEED)

# =============================================================================
# wait
# =============================================================================
while True:
    if os.path.isfile('SUCCESS_803'):
        break
    else:
        sleep(60*1)

utils.send_line('{} start'.format(__file__))
# =============================================================================
# load train
# =============================================================================

dtrain = xgb.DMatrix('../data/dtrain.mt')
gc.collect()

# =============================================================================
# xgboost
# =============================================================================

param = {'colsample_bylebel': 0.8,
         'subsample': 0.1,
         'eta': 0.1,
         'eval_metric': 'auc',
         'max_depth': 4,
         'objective': 'binary:logistic',
         'silent': 1,
         'tree_method': 'hist',
         'nthread': 64,
#         'seed': SEED,
         }

gc.collect()

cv = xgb.cv(param, dtrain, NROUND, nfold=5, early_stopping_rounds=50, verbose_eval=10)



