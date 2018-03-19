#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 18:28:14 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import xgboost as xgb
import utils
#utils.start(__file__)

# setting
SEED = 48

np.random.seed(SEED)

# =============================================================================
# load train & feature
# =============================================================================

train = pd.concat([utils.read_pickles('../data/train'),
                   pd.read_pickle('../data/101_train.p'),
                   pd.read_pickle('../data/102_train.p')]+[pd.read_pickle('../data/{}_train.p'.format('-'.join(keys))) for keys in utils.comb], 
                  axis=1)#.sample(frac=0.4)

gc.collect()


y = train.is_attributed
train.drop(['ip', 'app', 'device', 'os', 'channel', 'is_attributed', 'click_time', 'attributed_time'], 
           axis=1, inplace=True)
train.fillna(-1, inplace=True)

gc.collect()


# =============================================================================
# save
# =============================================================================

valid_seed = np.random.randint(99999)
X_valid = train.sample(frac=0.05, random_state=valid_seed)
y_valid = y.sample(frac=0.05, random_state=valid_seed)

dval = xgb.DMatrix(X_valid, y_valid)
dval.save_binary('../data/dval.mt')

train.drop(X_valid.index, inplace=True)
y = y.drop(X_valid.index)

del dval, X_valid, y_valid; gc.collect()

for i in range(10):
    train_seed = np.random.randint(99999)
    X_train = train.sample(frac=0.1, random_state=train_seed)
    y_train = y.sample(frac=0.1, random_state=train_seed)
    xgb.DMatrix(X_train, y_train).save_binary('../data/dtrain{}.mt'.format(i))
    
    gc.collect()



