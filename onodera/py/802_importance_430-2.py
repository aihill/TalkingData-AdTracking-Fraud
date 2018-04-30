#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:07:30 2018

@author: Kazuki
"""


import pandas as pd
import numpy as np
from os import system
import os
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
import lightgbm as lgb
import gc
from time import sleep
import utils
utils.start(__file__)

SEED = 71 #np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999

np.random.seed(SEED)
print('seed :', SEED)

system('rm SUCCESS_802')

categorical_feature = [ 'app', 'device', 'os', 'channel', 'hour']


param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.2,
         'max_depth': 4,
         'num_leaves': 2**4-1,
         'max_bin': 100,
         'min_child_samples': 300,
         'min_child_weight': 0,
         'colsample_bytree': 0.8,
         'subsample': 0.1,
         'nthread': 64,
         'scale_pos_weight': 100,
#         'lambda_l1': 3,
#         'lambda_l2': 3,
         }

# =============================================================================
# wait
# =============================================================================
while True:
    if os.path.isfile('SUCCESS_801'):
        break
    else:
        sleep(60*1)

utils.send_line('START {}'.format(__file__))

# =============================================================================
# load
# =============================================================================
imp = pd.read_csv('imp_802_importance_430-1.py.csv').set_index('index')
usecols = imp[imp.weight!=0].index.tolist()

X_train = pd.read_feather('../data/X_train_mini.f')[usecols]

gc.collect()

X_valid = pd.read_feather('../data/X_valid_mini.f')[usecols]
gc.collect()


drop_col = ['ip', 'app', 'device', 'os', 'channel', 'hour']
for drop_c in drop_col:
    # =============================================================================
    print(f'===================== LGB drop {drop_c} =====================')
    # =============================================================================
    categorical_feature_ = list( set(categorical_feature) - set([drop_c]))
    
    dtrain = lgb.Dataset(X_train.drop(drop_c, axis=1),
                         label=pd.read_feather('../data/y_train_mini.f').is_attributed,
                         categorical_feature=categorical_feature_)
    
    gc.collect()
    
    dvalid = lgb.Dataset(X_valid.drop(drop_c, axis=1),
                         label=pd.read_feather('../data/y_valid_mini.f').is_attributed,
                         categorical_feature=categorical_feature_)
    
    gc.collect()
    
    
    evals_result = {}
    
    np.random.seed(SEED)
    models = []
    for i in range(1):
        param.update({'seed':np.random.randint(9999)})
        model = lgb.train(params=param, train_set=dtrain, num_boost_round=NROUND, 
                          valid_sets=[dtrain, dvalid], 
                          valid_names=['train','valid'], 
                          early_stopping_rounds=50, 
                          evals_result=evals_result, 
                          verbose_eval=10,
                          categorical_feature=categorical_feature_
                          )
        models.append(model)
    
    imp = ex.getImp(models)
    
    #t = datetime.today()
    #date = t.date()
    #hour = t.hour
    imp.to_csv(f'imp_{__file__}_{drop_c}.csv', index=False)



#==============================================================================
system('touch SUCCESS_802')
utils.end(__file__)

