#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:04:05 2018

@author: Kazuki

sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import xgboost as xgb
import gc
import utils
utils.start(__file__)

seed = 71
np.random.seed(seed)

# =============================================================================
# load train
# =============================================================================

dbuild = xgb.DMatrix('../data/dbuild_10per.mt')
dvalid = xgb.DMatrix('../data/dvalid_10per.mt')

watchlist = [(dbuild, 'build'),(dvalid, 'valid')]
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


model = xgb.train(param, dbuild, 9999, watchlist,
                  early_stopping_rounds=50, verbose_eval=5)

# =============================================================================
# cv
# =============================================================================

#cv = xgb.cv(param, X, 9999, 
#            nfold=5, early_stopping_rounds=50, verbose_eval=5)

#==============================================================================
utils.end(__file__)
