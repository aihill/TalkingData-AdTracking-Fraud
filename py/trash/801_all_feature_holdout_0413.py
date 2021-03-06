#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 21:47:15 2018

@author: Kazuki
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
X = pd.concat([pd.read_pickle('../data/train/6.p'),
               pd.read_pickle('../data/002_train/6.p'),
               pd.read_pickle('../data/003_train/6.p'),
               pd.read_pickle('../data/004_train/6.p'),
               pd.read_pickle('../data/005_train/6.p'),
               pd.read_pickle('../data/006_train/6.p'),
               pd.read_pickle('../data/101_train/6.p'),
               pd.read_pickle('../data/102_train/6.p'),
               pd.read_pickle('../data/103-1_train/6.p'),
               pd.read_pickle('../data/103-2_train/6.p'),
               ], axis=1)
gc.collect()

y = X.is_attributed
X.drop(['ip', 'app', 'device', 'os', 'channel', 'is_attributed', 'click_time', 'attributed_time'], 
           axis=1, inplace=True)
X.fillna(-1, inplace=True)

gc.collect()

col = X.columns
print(col)

dtrain = xgb.DMatrix(X, y)

# =============================================================================
# load valid
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
               pd.read_pickle('../data/103-1_train/9.p'),
               pd.read_pickle('../data/103-2_train/9.p'),
               ], axis=1)
gc.collect()

y = X.is_attributed
X.drop(['ip', 'app', 'device', 'os', 'channel', 'is_attributed', 'click_time', 'attributed_time'], 
           axis=1, inplace=True)
X.fillna(-1, inplace=True)

gc.collect()

dvalid = xgb.DMatrix(X, y)


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


gc.collect()
watchlist = [(dtrain, 'train'),(dvalid, 'valid')]

# =============================================================================
# cv
# =============================================================================

model = xgb.train(param, dtrain, NROUND, watchlist, verbose_eval=10, 
                  early_stopping_rounds=50)

imp = ex.getImp(model)
imp.to_csv('imp.csv', index=False)


#==============================================================================
utils.end(__file__)


