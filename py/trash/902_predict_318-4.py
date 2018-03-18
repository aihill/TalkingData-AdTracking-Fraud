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
from itertools import combinations
import utils

# setting
submit_file_path = '../output/318-4.csv.gz'
SEED = 48
LOOP = 3
nround = 300

np.random.seed(SEED)
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
# XGBoost
# =============================================================================
param = {'colsample_bylebel': 0.8,
         'subsample': 0.5,
         'eta': 0.1,
         'eval_metric': 'auc',
         'max_depth': 6,
         'objective': 'binary:logistic',
         'silent': 1,
         'tree_method': 'hist',
         'nthread': 64}

print('train.shape:', train.shape)
train_head = train.head()
train_head.to_pickle('train_head.p')

dtrain = xgb.DMatrix(train, y)
del train, y; gc.collect()

print('start xgb')
models = []
for i in range(LOOP):
    print(i)
    param.update({'seed':np.random.randint(9999)})
    model = xgb.train(param, dtrain, nround)
    model.save_model('xgb{}.model'.format(i))
    models.append(model)

imp = ex.getImp(models)
imp.to_csv('imp_{}.csv'.format(__file__), index=False)

del dtrain; gc.collect()

# =============================================================================
# test
# =============================================================================
# feature
test = utils.read_pickles('../data/test')

print('test.shape:', test.shape) # 18790469

gc.collect()

for keys in tqdm(comb):
    gc.collect()
    keys_ = '-'.join(keys)
    df = pd.read_pickle('../data/{}_count_old.p'.format(keys_))
    test = pd.merge(test, df, on=keys, how='left')



sub = test[['click_id']]

test.drop(['click_id', 'ip', 'app', 'device', 'os', 'channel', 'click_time'], 
           axis=1, inplace=True)
test.fillna(-1, inplace=True)

gc.collect()

print(test.columns.tolist())


test.fillna(-1, inplace=True)

dtest = xgb.DMatrix(test[train_head.columns])

sub['is_attributed'] = 0
for model in models:
    y_pred = model.predict(dtest)
    sub['is_attributed'] += pd.Series(y_pred).rank()
sub['is_attributed'] /= LOOP
sub['is_attributed'] /= sub['is_attributed'].max()
sub['click_id'] = sub.click_id.map(int)

sub.to_csv(submit_file_path, index=False, compression='gzip')

# =============================================================================
# submission
# =============================================================================
utils.submit(submit_file_path)







