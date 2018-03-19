#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:04:05 2018

@author: Kazuki

sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"

nohup python -u 901_cv.py > log.txt &

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/home/kazuki_onodera/Python')
import xgbextension as ex
import xgboost as xgb
import gc
from itertools import combinations
import utils

seed = 71
np.random.seed(seed)

# =============================================================================
# load train
# =============================================================================

train = utils.read_pickles('../data/train').sample(frac=0.1)

gc.collect()

comb = list(combinations(['ip', 'app', 'device', 'os', 'channel'], 4))
comb += list(combinations(['ip', 'app', 'device', 'os', 'channel'], 3))
comb += list(combinations(['ip', 'app', 'device', 'os', 'channel'], 2))

for keys in tqdm(comb):
    gc.collect()
    keys_ = '-'.join(keys)
    df = pd.read_pickle('../data/{}_count_old.p'.format(keys_))
    train = pd.merge(train, df, on=keys, how='left')

train.drop(['click_time', 'attributed_time'], axis=1, inplace=True)

y = train.is_attributed
train.drop(['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 
           axis=1, inplace=True)
train.fillna(-1, inplace=True)

gc.collect()

print(train.columns.tolist())

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
         'nthread': 64,
         'seed':71}


gc.collect()


#yhat, imp, ret = ex.stacking(train, y, param, 9999, nfold=5, esr=30)
#
#imp.to_csv('imp.csv', index=False)

# =============================================================================
# cv
# =============================================================================

dtrain = xgb.DMatrix(train, y)

cv = xgb.cv(param, dtrain, 9999, 
            nfold=5, early_stopping_rounds=50, verbose_eval=5)



