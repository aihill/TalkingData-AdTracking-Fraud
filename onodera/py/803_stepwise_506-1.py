#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 23:42:58 2018

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
from itertools import combinations
from time import sleep
import utils
utils.start(__file__)

TARGET_SEED = 3833

SEED = 71 #np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999

np.random.seed(SEED)
print('seed :', SEED)

system('rm SUCCESS_803')

categorical_feature = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']
categorical_feature += ['nearestNext_ip', 'nearestPre_ip', 'nearestNext_app', 
                        'nearestPre_app', 'nearestNext_device', 'nearestPre_device', 
                        'nearestNext_os', 'nearestPre_os', 'nearestNext_channel',
                        'nearestPre_channel']


param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.1,
         'max_depth': 3, # 4?
         'num_leaves': 2**3-1,
         'max_bin': 100,
         'min_child_samples': 300,
         'min_child_weight': 100,
         'colsample_bytree': 0.7,
         'subsample': 0.5,
         'nthread': 64,
         'scale_pos_weight': 100,
         'bagging_freq': 1,
#         'lambda_l1': 3,
#         'lambda_l2': 3,
         
         'seed': SEED
         }

# =============================================================================
# wait
# =============================================================================
while True:
    if os.path.isfile('SUCCESS_802'):
        break
    else:
        sleep(60*1)

utils.send_line('START {}'.format(__file__))

# =============================================================================
# load
# =============================================================================
imp = pd.read_csv('imp_802_importance_506-1.py.csv').set_index('index')
feature_all = imp[imp.weight!=0].index.tolist()

X_train = pd.read_feather(f'../data/X_train_mini_s{TARGET_SEED}.f')[feature_all]
y_train = pd.read_feather(f'../data/y_train_mini_s{TARGET_SEED}.f').is_attributed

gc.collect()

X_valid = pd.read_feather(f'../data/X_valid_mini_s{TARGET_SEED}.f')[feature_all]
y_valid = pd.read_feather(f'../data/y_valid_mini_s{TARGET_SEED}.f').is_attributed
gc.collect()

# =============================================================================
# def
# =============================================================================

def do_lgb(features):
    categorical_feature_ = list( set(categorical_feature) & set(features) )
    
    dtrain = lgb.Dataset(X_train[features], label=y_train,
                         categorical_feature=categorical_feature_)
    
    gc.collect()
    
    dvalid = lgb.Dataset(X_valid[features], label=y_valid,
                         categorical_feature=categorical_feature_)
    evals_result = {}
    gc.collect()
    
    model = lgb.train(params=param, train_set=dtrain, num_boost_round=NROUND, 
                      valid_sets=[dtrain, dvalid], 
                      valid_names=['train', 'valid'], 
                      early_stopping_rounds=50, 
                      evals_result=evals_result, 
                      verbose_eval=False,
#                      categorical_feature=categorical_feature_
                      )
    
    train_score = model.best_score['train']['auc']
    valid_score = model.best_score['valid']['auc']
    
    print(f'train auc:{train_score}  valid auc:{valid_score}')
    
    return valid_score



# =============================================================================
# main
# =============================================================================
use_features = list( set(utils.BEST_FEATURES_504) & set(feature_all) )

best_score = do_lgb(use_features)
print(f'benchmark {best_score}')

max_feature_length = 100
index = 0
drop_features = []
use_features_bk = []

while True:
    
    for feature in feature_all:
        
        use_features2 = use_features[:]
        if feature in use_features2:
            use_features2.remove(feature)
        else:
            use_features2.append(feature)
        
        print(f'\n\nTRY to USE {feature_all.index(feature)} {use_features2}')
        
        score = do_lgb(use_features2)
        
        diff_score = score - best_score
        if best_score < score:
            best_score = score
            use_features = use_features2
            print(f'UPDATE!    DIFF:{diff_score:+.5f}    SCORE:{best_score:+.5f}')
        else:
            print(f'Failed.    DIFF:{diff_score:+.5f}    SCORE:{best_score:+.5f}')
    
        gc.collect()
    
    if len(use_features) >= max_feature_length:
        print(f'break! coz len(use_features) >= max_feature_length')
        break
    elif use_features == use_features_bk:
        print(f'break! coz cannt improve')
        break
    
    use_features_bk = use_features[:]

"""

"""




#==============================================================================
system('touch SUCCESS_803')
utils.end(__file__)


