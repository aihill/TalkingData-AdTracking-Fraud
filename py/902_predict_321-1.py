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
from multiprocessing import Pool
from time import sleep
import utils
utils.start(__file__)

# setting
SUBMIT_FILE_PATH = '../output/321-1.csv.gz'
SEED = 48
TOTAL_NROUND = 50
EACH_NROUND = 5
EXE_SUBMIT = True



np.random.seed(SEED)

# =============================================================================
# def
# =============================================================================
def multi(arg):
    'plan1: ValueError: ctypes objects containing pointers cannot be pickled'
#    if arg is None:
#        "load data"
#        load_file = '../data/dtrain{}.mt'.format(np.random.randint(10))
#        return xgb.DMatrix(load_file)
#    
#    elif isinstance(arg, list):
#        "train"
#        return xgb.train(param, arg[0], EACH_NROUND, xgb_model=arg[1])
#    
#    else:
#        raise Exception(arg)
    
    'plan2: NameError: name "model_new" is not defined'
#    if arg==0:
#        "load data"
#        sleep(10) # delay for train
#        global dtrain_new
#        load_file = '../data/dtrain{}.mt'.format(np.random.randint(10))
#        print(load_file)
#        dtrain_new = xgb.DMatrix(load_file)
#        return 
#    
#    elif arg==1:
#        "train"
#        global model_new
#        model_new = xgb.train(param, dtrain, EACH_NROUND, xgb_model=model)
#        return
#    
#    else:
#        raise Exception(arg)

    'plan3: "Reason: error(i format requires -2147483648 <= number <= 2147483647,)"'
#    if arg==0:
#        "load data"
#        n = np.random.randint(10)
#        load_file = '../data/X_train{}.p'.format(n)
#        print(load_file)
#        X_train = pd.read_pickle(load_file)
#        load_file = '../data/y_train{}.p'.format(n)
#        y_train = pd.read_pickle(load_file)
#        return X_train, y_train
#    
#    elif arg==1:
#        "train"
#        return xgb.train(param, dtrain, EACH_NROUND, xgb_model=model)
#    
#    else:
#        raise Exception(arg)


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
    
    # for multi
#    pool = Pool(2)
#    callback = pool.map(multi, [0, 1])
#    pool.close()
#    callback
    
    load_file = '../data/dtrain{}.mt'.format(np.random.randint(10))
    model = xgb.train(param, xgb.DMatrix(load_file), EACH_NROUND, xgb_model=model)
    
    auc = roc_auc_score(dvalid.get_label(), model.predict(dvalid))
    current_nround += EACH_NROUND
    print('[NROUND]: {}    [valid-auc]: {:.7f}    {:.2f} min'.format(auc, current_nround, utils.elapsed_minute()))
    if current_nround >= TOTAL_NROUND:
        break

imp = ex.getImp(model)
imp.to_csv('LOG/imp_{}.csv'.format(__file__), index=False)


# =============================================================================
# test
# =============================================================================
sub = pd.read_pickle('../data/sub.p').reset_index()

dtest = xgb.DMatrix('../data/dtest.mt')

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




