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

categorical_feature = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']

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
dtrain = lgb.Dataset(pd.read_feather('../data/X_train.f'),
                     label=pd.read_feather('../data/y_train.f').is_attributed,
                     categorical_feature=categorical_feature)

gc.collect()


dvalid = lgb.Dataset(pd.read_feather('../data/X_valid.f'),
                     label=pd.read_feather('../data/y_valid.f').is_attributed,
                     categorical_feature=categorical_feature)
gc.collect()

# =============================================================================
# lgb
# =============================================================================

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
         
         'seed': SEED
         }

gc.collect()

evals_result = {}

models = []
for i in range(3):
    param.update({'seed':np.random.randint(9999)})
    model = lgb.train(params=param, train_set=dtrain, num_boost_round=NROUND, 
                      valid_sets=[dtrain, dvalid], 
                      valid_names=['train','valid'], 
                      early_stopping_rounds=50, 
                      evals_result=evals_result, 
                      verbose_eval=10,
                      categorical_feature=categorical_feature
                      )
    models.append(model)

imp = ex.getImp(models)

#t = datetime.today()
#date = t.date()
#hour = t.hour
imp.to_csv(f'imp_{__file__}.csv', index=False)


system('touch SUCCESS_802')



#==============================================================================
utils.end(__file__)

