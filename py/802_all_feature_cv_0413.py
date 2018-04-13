#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:04:09 2018

@author: kazuki.onodera
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys
sys.path.append('/home/kazuki_onodera/Python')
import xgbextension as ex
#import xgboost as xgb
#from multiprocessing import Process, Pipe
import gc
import xgboost as xgb
#from time import sleep
import utils
utils.start(__file__)

SEED = 4308 # np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999


np.random.seed(SEED)
print('seed :', SEED)
# =============================================================================
# load train
# =============================================================================


# by datetime dvalid
X = pd.concat([pd.read_pickle('../data/train/9.p'),
               pd.read_pickle('../data/002_train/9.p'),
               pd.read_pickle('../data/003_train/9.p'),
               pd.read_pickle('../data/004_train/9.p'),
               pd.read_pickle('../data/005_train/9.p'),
               pd.read_pickle('../data/006_train/9.p'),
               pd.read_pickle('../data/101_train/9.p'),
               pd.read_pickle('../data/102_train/9.p'),
               pd.read_pickle('../data/103_train/9.p'),
               ], axis=1)
gc.collect()

y = X.is_attributed
X.drop(['ip', 'app', 'device', 'os', 'channel', 'is_attributed', 'click_time', 'attributed_time'], 
           axis=1, inplace=True)
X.fillna(-1, inplace=True)

gc.collect()

col = X.columns
print(col)

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
         'seed': SEED}



#dtrain = xgb.DMatrix(X, y)
#col = X.columns
#del X, y; gc.collect()

gc.collect()
yhat, imp, ret = ex.stacking(X, y, param, 9999, nfold=5, esr=30)

imp.to_csv('imp_{}.csv'.format(datetime.today().date()), index=False)

# =============================================================================
# cv
# =============================================================================

#model = xgb.train(param, dbuild, NROUND, watchlist, verbose_eval=10, 
#                  early_stopping_rounds=50)
#
#imp = ex.getImp(model)
#imp.to_csv('imp.csv', index=False)



#==============================================================================
utils.end(__file__)

