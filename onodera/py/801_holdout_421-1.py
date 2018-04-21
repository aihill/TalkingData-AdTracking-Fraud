#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:27:08 2018

@author: kazuki.onodera
"""

import pandas as pd
import numpy as np
from os import system
from datetime import datetime
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
#import lightgbm as lgb
import gc
#from time import sleep
import utils
utils.start(__file__)

SEED = np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999


np.random.seed(SEED)
print('seed :', SEED)
system('rm SUCCESS_801')

utils.send_line('START {}'.format(__file__))

# =============================================================================
# load train
# =============================================================================


# by datetime dvalid
X = pd.concat([pd.read_pickle('../data/train/9.p').tail(9184919),
               pd.read_pickle('../data/001_train/9.p').tail(9184919),
               pd.read_pickle('../data/002_train/9.p').tail(9184919),
               pd.read_pickle('../data/003_train/9.p').tail(9184919),
               pd.read_pickle('../data/004_train/9.p').tail(9184919),
               pd.read_pickle('../data/101_train/9.p').tail(9184919),
               pd.read_pickle('../data/102_train/9.p').tail(9184919),
               pd.read_pickle('../data/103-1_train/9.p').tail(9184919),
               pd.read_pickle('../data/103-2_train/9.p').tail(9184919),
               pd.read_pickle('../data/103-3_train/9.p').tail(9184919),
               pd.read_pickle('../data/104-1_train/9.p').tail(9184919),
               pd.read_pickle('../data/104-2_train/9.p').tail(9184919),
               pd.read_pickle('../data/104-3_train/9.p').tail(9184919),
               pd.read_pickle('../data/105-1_train/9.p').tail(9184919),
               pd.read_pickle('../data/105-2_train/9.p').tail(9184919),
               pd.read_pickle('../data/106-1_train/9.p').tail(9184919),
               pd.read_pickle('../data/106-2_train/9.p').tail(9184919),
               pd.read_pickle('../data/106-3_train/9.p').tail(9184919),
               pd.read_pickle('../data/107-1_train/9.p').tail(9184919),
               pd.read_pickle('../data/107-2_train/9.p').tail(9184919),
               pd.read_pickle('../data/108-1_train/9.p').tail(9184919),
               pd.read_pickle('../data/108-2_train/9.p').tail(9184919),
               pd.read_pickle('../data/108-3_train/9.p').tail(9184919),
               pd.read_pickle('../data/109-1_train/9.p').tail(9184919),
               pd.read_pickle('../data/109-2_train/9.p').tail(9184919),
               pd.read_pickle('../data/109-3_train/9.p').tail(9184919),
               pd.read_pickle('../data/109-4_train/9.p').tail(9184919),
               pd.read_pickle('../data/109-5_train/9.p').tail(9184919),
               pd.read_pickle('../data/109-6_train/9.p').tail(9184919),
               pd.read_pickle('../data/109-7_train/9.p').tail(9184919),
               pd.read_pickle('../data/109-8_train/9.p').tail(9184919),
               pd.read_pickle('../data/109-9_train/9.p').tail(9184919),
               pd.read_pickle('../data/110-1_train/9.p').tail(9184919),
               pd.read_pickle('../data/110-2_train/9.p').tail(9184919),
               pd.read_pickle('../data/110-3_train/9.p').tail(9184919),
               pd.read_pickle('../data/110-4_train/9.p').tail(9184919),
               pd.read_pickle('../data/111-1_train/9.p').tail(9184919),
               pd.read_pickle('../data/111-2_train/9.p').tail(9184919),
               pd.read_pickle('../data/111-3_train/9.p').tail(9184919),
               ], axis=1)
gc.collect()

y = X.is_attributed
categorical_feature = ['app', 'device', 'os', 'channel', 'day', 'hour']
drop_feature = ['ip', 'is_attributed', 'click_time', 'attributed_time']
X.drop(drop_feature, axis=1, inplace=True)
X.fillna(-1, inplace=True)

gc.collect()

col = X.columns
print(col)

# =============================================================================
# lgbm
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
         'seed': SEED
         }

gc.collect()
yhat, imp, ret = ex.stacking(X, y, param, NROUND, nfold=5, esr=50, 
                             categorical_feature=categorical_feature)

t = datetime.today()
date = t.date()
hour = t.hour
imp.to_csv('imp_{}-{:02d}h.csv'.format(date, hour), index=False)



system('touch SUCCESS_801')



#==============================================================================
utils.end(__file__)
