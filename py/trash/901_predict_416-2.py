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
import os
#from multiprocessing import Process, Pipe
import gc
import xgboost as xgb
from time import sleep
import utils
utils.start(__file__)

SEED = 4308 # np.random.randint(9999) #int(sys.argv[1])
SUBMIT_FILE_PATH = '../output/416-2.csv.gz'
EXE_SUBMIT = True
LOOP = 5

np.random.seed(SEED)
print('seed :', SEED)

# =============================================================================
# wait
# =============================================================================
while True:
    if os.path.isfile('SUCCESS'):
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

cv = xgb.cv(param, dtrain, 9999, nfold=5, early_stopping_rounds=50, verbose_eval=10)
gc.collect()
NROUND = int(cv.shape[0] * 1.2)

models = []
for i in range(LOOP):
    gc.collect()
    param.update({'seed':np.random.randint(9999)})
    model = xgb.train(param, dtrain, NROUND)
    models.append(model)
    
del dtrain; gc.collect()

imp = ex.getImp(models)
imp.to_csv('901_imp.csv', index=False)

# =============================================================================
# test
# =============================================================================

sub = pd.read_pickle('../data/sub.p')
dtest = xgb.DMatrix('../data/dtest.mt')
gc.collect()

sub['is_attributed'] = 0
for model in models:
    y_pred = model.predict(dtest)
    sub['is_attributed'] += pd.Series(y_pred).rank()
sub['is_attributed'] /= LOOP
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

