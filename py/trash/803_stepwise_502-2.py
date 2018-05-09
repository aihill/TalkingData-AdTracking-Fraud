#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:16:41 2018

@author: kazuki.onodera
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

SEED = 71 #np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999

np.random.seed(SEED)
print('seed :', SEED)

system('rm SUCCESS_803')

categorical_feature = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']


param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.1,
         'max_depth': 3, # 4?
         'num_leaves': 2**3-1,
         'max_bin': 100,
         'min_child_samples': 300,
         'min_child_weight': 0,
         'colsample_bytree': 1,
         'subsample': 1,
         'nthread': 64,
         'scale_pos_weight': 100,
#         'lambda_l1': 3,
#         'lambda_l2': 3,
         
         'seed': SEED
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
imp = pd.read_csv('imp_802_importance_502-2.py.csv').set_index('index')
feature_all = imp[imp.weight!=0].index.tolist()

X_train = pd.read_feather('../data/X_train_mini.f')[feature_all]
y_train = pd.read_feather('../data/y_train_mini.f').is_attributed

gc.collect()

X_valid = pd.read_feather('../data/X_valid_mini.f')[feature_all]
y_valid = pd.read_feather('../data/y_valid_mini.f').is_attributed
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

best_score = do_lgb(feature_all[:5])
print(f'benchmark {best_score}')

max_feature_length = 100
index = 0
drop_features = []
use_features = feature_all[:5]
use_features_bk = []

while True:
    
    for feature in feature_all[5:]:
        
#        drop_features.append(feature)
#        use_features = list(set(X_train.columns) - set(drop_features))
#        print(f'\nTRY to DROP {drop_features}')
        
        use_features.append(feature)
        print(f'\n\nTRY to USE {use_features}')
        
        score = do_lgb(use_features)
        
        diff_score = score - best_score
        if best_score < score:
            best_score = score
            print(f'UPDATE!    DIFF:{diff_score:+.5f}    SCORE:{best_score:+.5f}')
        else:
#            drop_features.remove(feature)
            use_features.remove(feature)
            print(f'Failed.    DIFF:{diff_score:+.5f}    SCORE:{best_score:+.5f}')
    
        gc.collect()
    
    if len(use_features) >= max_feature_length:
        print(f'break! coz len(use_features) >= max_feature_length')
        break
    elif use_features == use_features_bk:
        print(f'break! coz cannt improve')
        break
    break
    use_features_bk = use_features[:]

"""

['ip', 'channel', 'app', 'os', 'timedelta_rev_ip-app-device-os', 
'timedelta_rev_ip-app-os', 'hour_min', 'device', 'timeDD_rev_ip-app-device-os', 
'nunique_ip-app-device-channel_ip-device', 'nunique_ip-app_ip', 
'nunique_ip-app-device-os_ip-os', 'nunique_ip-os_ip', 'timediff-minmax_app-channel', 
'totalcount_app', 'nunique_app-device-os-channel_app-channel', 'timeskew_ip', 
'nunique_device-os-channel_channel', 'totalcount_ip-app', 
'nunique_ip-app-device-channel_ip-channel', 'timevar_app-os-channel', 
'timedelta2_app-device-os-channel', 'timediff-meadian_ip-app-device-channel', 
'timeDD_device-os', 'timeDD_rev_ip-app-device-os-channel']


"""




#==============================================================================
system('touch SUCCESS_803')
utils.end(__file__)


