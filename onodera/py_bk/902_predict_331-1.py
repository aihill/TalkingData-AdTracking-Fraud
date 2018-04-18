#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:02:42 2018

@author: kazuki.onodera
"""
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/kazuki_onodera/Python')
import xgbextension as ex
import xgboost as xgb
import gc
from sklearn.metrics import roc_auc_score
#from multiprocessing import Process, Pipe
from threading import Thread
from queue import Queue
from time import sleep
import utils
utils.start(__file__)

# setting
SUBMIT_FILE_PATH = '../output/331-1.csv.gz'
SEED = np.random.randint(9999)
TOTAL_NROUND = 2500
EACH_NROUND = 30
EXE_SUBMIT = True


dmatrix_queue = Queue()
np.random.seed(SEED)
print('seed:', SEED)
# =============================================================================
# def
# =============================================================================

def sender(load_file):
#    print('loading {} ...'.format(load_file))
    dmatrix_queue.put(xgb.DMatrix(load_file))
    
# =============================================================================
# XGBoost
# =============================================================================
param = {'colsample_bylebel': 0.8,
         'subsample': 0.8,
         'eta': 0.1,
         'eval_metric': 'auc',
         'max_depth': 4,
         'objective': 'binary:logistic',
         'silent': 1,
         'tree_method': 'hist',
         'nthread': 64,
         'seed':71}

dvalid = xgb.DMatrix('../data/dvalid_10per.mt')

print('start xgb')
model = None
current_nround = 0
auc_best = 0
auc_decrease = 0

load_files = glob('../data/dtrain*.mt')
np.random.shuffle(load_files)
load_file_len = len(load_files)
load_file_index = 0
load_file = load_files[load_file_index]
load_file_index +=1
Thread(target=sender, args=(load_file, )).start()
dtrain = dmatrix_queue.get()


while True:
    gc.collect()
    param.update({'seed':np.random.randint(9999)})
    
    load_file = load_files[load_file_index]
    load_file_index +=1
    if load_file_index >= load_file_len:
        load_file_index = 0
    Thread(target=sender, args=(load_file, )).start()
    model = xgb.train(param, dtrain, EACH_NROUND, xgb_model=model)
    dtrain = dmatrix_queue.get()
    
    auc = roc_auc_score(dvalid.get_label(), model.predict(dvalid))
    current_nround += EACH_NROUND
    print('[NROUND]: {}    [valid-auc]: {:.7f}    {:.2f} min'.format(current_nround, auc, utils.elapsed_minute()))
    if current_nround >= TOTAL_NROUND :
        break

imp = ex.getImp(model)
imp.to_csv('LOG/imp_{}.csv'.format(__file__), index=False)


# =============================================================================
# test
# =============================================================================
sub = pd.read_pickle('../data/sub.p').reset_index(drop=True)

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




