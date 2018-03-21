#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:02:42 2018

@author: kazuki.onodera
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/kazuki_onodera/Python')
import xgbextension as ex
import xgboost as xgb
import gc
from sklearn.metrics import roc_auc_score
import utils
utils.start(__file__)

# setting
SUBMIT_FILE_PATH = '../output/321-1.csv.gz'
SEED = 48
TOTAL_NROUND = 800
EACH_NROUND = 5
EXE_SUBMIT = True



np.random.seed(SEED)

# =============================================================================
# XGBoost
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

dvalid = xgb.DMatrix('../data/dvalid.mt')

print('start xgb')
model = None
current_nround = 0
while True:
    gc.collect()
    param.update({'seed':np.random.randint(9999)})
    load_file = '../data/dtrain{}.mt'.format(np.random.randint(10))
    model = xgb.train(param, xgb.DMatrix(load_file), EACH_NROUND, xgb_model=model)
    auc = roc_auc_score(dvalid.get_label(), model.predict(dvalid))
    current_nround += EACH_NROUND
    print('valid-auc: {} NROUND {} Done. {} min'.format(auc, current_nround, utils.elapsed_minute()))
    if current_nround >= TOTAL_NROUND:
        break

imp = ex.getImp(model)
imp.to_csv('LOG/imp_{}.csv'.format(__file__), index=False)


# =============================================================================
# test
# =============================================================================
# feature
test = pd.concat([utils.read_pickles('../data/test_old'),
                   pd.read_pickle('../data/101_test_old.p'),
                   pd.read_pickle('../data/102_test_old.p')]+[pd.read_pickle('../data/{}_test.p'.format('-'.join(keys))) for keys in utils.comb], 
                  axis=1)#.sample(frac=0.4)

gc.collect()

test = test[~test.click_id.isnull()]
test.drop_duplicates('click_id', keep='last', inplace=True) # last?

print('test.shape should be 18790469:', test.shape)

gc.collect()

sub = test[['click_id']]

test.drop(['click_id', 'ip', 'app', 'device', 'os', 'channel', 'click_time'], 
           axis=1, inplace=True)
test.fillna(-1, inplace=True)

gc.collect()

print(test.columns.tolist())

test.fillna(-1, inplace=True)

train_head = pd.read_pickle('train_head.p')
dtest = xgb.DMatrix(test[train_head.columns])

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




