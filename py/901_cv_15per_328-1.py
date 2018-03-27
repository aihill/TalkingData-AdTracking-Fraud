#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:04:05 2018

@author: Kazuki

sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"

%time X = xgb.DMatrix('../data/dvalid_15per.mt')
[22:10:13] 27735584x100 matrix with 2773558400 entries loaded from ../data/dvalid_15per.mt
CPU times: user 105 ms, sys: 28.7 s, total: 28.8 s
Wall time: 2min 36s

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/home/kazuki_onodera/Python')
import xgbextension as ex
import xgboost as xgb
import gc
import utils
utils.start(__file__)

seed = 71
np.random.seed(seed)

# =============================================================================
# load train
# =============================================================================

X = xgb.DMatrix('../data/dvalid_15per.mt')

# =============================================================================
# xgboost
# =============================================================================


param = {'colsample_bylebel': 0.8,
         'subsample': 0.5,
         'eta': 0.1,
         'eval_metric': 'auc',
         'max_depth': 4,
         'objective': 'binary:logistic',
         'silent': 1,
         'tree_method': 'hist',
         'nthread': 30,
         'seed':seed}


gc.collect()


#yhat, imp, ret = ex.stacking(train, y, param, 9999, nfold=5, esr=30)
#
#imp.to_csv('imp.csv', index=False)

# =============================================================================
# cv
# =============================================================================

cv = xgb.cv(param, X, 9999, 
            nfold=5, early_stopping_rounds=50, verbose_eval=5)

#==============================================================================
utils.end(__file__)

