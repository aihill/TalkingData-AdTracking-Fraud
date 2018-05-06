#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 10:48:02 2018

@author: Kazuki
"""


import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lightgbm as lgb
import os
import gc
from time import sleep
import utils
utils.start(__file__)

SEED = np.random.randint(9999) #int(sys.argv[1])
NROUND = 400
LOOP = 3
SUBMIT_FILE_PATH = '../output/506-3.csv.gz'
COMMENT = f"r{NROUND}"

EXE_SUBMIT = True

np.random.seed(SEED)
print('seed :', SEED)

# =============================================================================
# wait
# =============================================================================
while True:
    if os.path.isfile('SUCCESS_805'):
        break
    else:
        sleep(60*1)

utils.send_line(f'START {__file__}')
# =============================================================================
# load train
# =============================================================================

dtrain = lgb.Dataset('../data/dtrain.mt')
gc.collect()


# =============================================================================
# xgboost
# =============================================================================

param = {
         'objective': 'binary',
         'metric': 'auc',
         'max_bin': 100,
         'nthread': 64,
         
         'learning_rate': 0.1,
         'max_depth': 4,
         'num_leaves': 2**4-1,
         'min_child_samples': 300,
         'min_child_weight': 100,
         'scale_pos_weight': 200,
         'colsample_bytree': 0.8,
         'subsample': 0.1,
         'bagging_freq': 5,
         'seed': SEED}

gc.collect()

models = []
for i in range(LOOP):
    gc.collect()
    param.update({'seed':np.random.randint(9999)})
    model = lgb.train(param, dtrain, NROUND)
    model.save_model(f'lgb{i}.model')
    models.append(model)
    
del dtrain; gc.collect()
"""

models = []
for i in range(3):
    bst = lgb.Booster(model_file=f'lgb{i}.model')
    models.append(bst)

imp = ex.getImp(models)

"""

# =============================================================================
# test
# =============================================================================

sub = pd.read_pickle('../data/sub.p')
dtest = utils.read_pickles('../data/dtest')
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
    print('submit')
    utils.submit(SUBMIT_FILE_PATH, COMMENT)


#==============================================================================
utils.end(__file__)


