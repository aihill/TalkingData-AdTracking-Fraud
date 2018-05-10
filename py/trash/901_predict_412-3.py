#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 01:20:46 2018

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
NROUND = 300
FRAC = 0.3
SUBMIT_FILE_PATH = '../output/412-3.csv.gz'
EXE_SUBMIT = True


np.random.seed(SEED)
print('seed :', SEED)
# =============================================================================
# load train
# =============================================================================

X = pd.concat([utils.read_pickles('../data/train').sample(frac=FRAC, random_state=SEED),
               utils.read_pickles('../data/002_train').sample(frac=FRAC, random_state=SEED),
               utils.read_pickles('../data/003_train').sample(frac=FRAC, random_state=SEED),
               utils.read_pickles('../data/004_train').sample(frac=FRAC, random_state=SEED),
               utils.read_pickles('../data/005_train').sample(frac=FRAC, random_state=SEED),
               utils.read_pickles('../data/006_train').sample(frac=FRAC, random_state=SEED),
               utils.read_pickles('../data/101_train').sample(frac=FRAC, random_state=SEED),
               ], axis=1)
gc.collect()

y = X.is_attributed
X.drop(['ip', 'app', 'device', 'os', 'channel', 'is_attributed', 'click_time', 'attributed_time'], 
           axis=1, inplace=True)
X.fillna(-1, inplace=True)
train_head = X.head()
train_head.to_pickle('train_head.p')


dtrain = xgb.DMatrix(X, y)
del X, y; gc.collect()


# =============================================================================
# xgboost
# =============================================================================

param = {'colsample_bylebel': 0.8,
         'subsample': 0.1,
         'eta': 0.01,
         'eval_metric': 'auc',
         'max_depth': 4,
         'objective': 'binary:logistic',
         'silent': 1,
         'tree_method': 'hist',
         'nthread': 64,
         'seed': SEED}


gc.collect()


model = xgb.train(param, dtrain, NROUND)
del dtrain; gc.collect()

imp = ex.getImp(model)
imp.to_csv('imp.csv', index=False)

# =============================================================================
# test
# =============================================================================

X = pd.concat([utils.read_pickles('../data/test_old'),
               utils.read_pickles('../data/002_test'),
               utils.read_pickles('../data/003_test'),
               utils.read_pickles('../data/004_test'),
               utils.read_pickles('../data/005_test'),
               utils.read_pickles('../data/006_test'),
               utils.read_pickles('../data/101_test'),
               ], axis=1)
gc.collect()

X = X[~X.click_id.isnull()]
X.drop_duplicates('click_id', keep='last', inplace=True) # last?
X.reset_index(drop=True, inplace=True)

print('test.shape should be 18790469:', X.shape)


sub = X[['click_id']]
sub.click_id = sub.click_id.map(int)

X.drop('click_id', axis=1, inplace=True)
X.fillna(-1, inplace=True)


dtest = xgb.DMatrix(X[train_head.columns])
del X; gc.collect()

sub['is_attributed'] = 0
y_pred = model.predict(dtest)
sub['is_attributed'] += pd.Series(y_pred).rank()
#sub['is_attributed'] /= LOOP
sub['is_attributed'] /= sub['is_attributed'].max()
sub['click_id'] = sub.click_id.map(int)

sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

# =============================================================================
# submission
# =============================================================================
if EXE_SUBMIT:
    utils.submit(SUBMIT_FILE_PATH)


#==============================================================================
utils.end(__file__)
