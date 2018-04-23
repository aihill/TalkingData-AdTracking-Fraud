#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 07:16:09 2018

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
NROUND = 500
LOOP = 3
SUBMIT_FILE_PATH = '../output/423-1.csv.gz'
COMMENT = F"r{NROUND} ['app', 'channel', 'dayvar_app-channel', 'dayvar_app-device', 'dayvar_app-device-os', 'dayvar_app-os-channel', 'hour', 'ip', 'is_attributed', 'nunique_app-device-channel_app', 'nunique_app-device-os_app-device', 'nunique_app-device-os_device-os', 'nunique_app-os-channel_app-channel', 'nunique_app-os-day_app', 'nunique_app-os-hour_app-hour', 'nunique_ip-app-channel_app-channel', 'nunique_ip-app-channel_ip-app', 'nunique_ip-app-day_ip', 'nunique_ip-app-device-channel_ip-app', 'nunique_ip-app-device-os_app-device', 'nunique_ip-app-device-os_ip-device-os', 'nunique_ip-app-device-os_ip-os', 'nunique_ip-app-device_ip-device', 'nunique_ip-app-hour_app-hour', 'nunique_ip-app-hour_ip', 'nunique_ip-app-hour_ip-hour', 'nunique_ip-app-os_app', 'nunique_ip-app-os_ip-os', 'nunique_ip-app_ip', 'nunique_ip-channel-day_ip-day', 'nunique_ip-channel_ip', 'nunique_ip-device-os_ip-os', 'nunique_ip-device_ip', 'nunique_ip-os-day_ip-day', 'nunique_ip-os-hour_os-hour', 'os', 'pre_att_time_app-os', 'pre_att_time_app-os-channel', 'pre_att_time_ip', 'pre_att_time_ip-app-device', 'pre_att_time_ip-app-os', 'timedelta_rev_ip', 'timedelta_rev_ip-app', 'timedelta_rev_ip-app-device', 'timedelta_rev_ip-app-device-os', 'timedelta_rev_ip-app-device-os-channel', 'timedelta_rev_ip-app-os', 'timedelta_rev_ip-app-os-channel', 'timediff-meadian_app', 'timediff-minmax_app-channel', 'timemean_app-device-os', 'timeskew_ip-device', 'timeskew_ip-os', 'timevar_device-os', 'totalcount_app', 'totalcount_app-channel', 'totalcount_app-device', 'totalcount_device-os', 'totalcount_ip', 'totalcount_ip-device', 'totalcount_ip-device-os']"

EXE_SUBMIT = True

np.random.seed(SEED)
print('seed :', SEED)

# =============================================================================
# wait
# =============================================================================
while True:
    if os.path.isfile('SUCCESS_802'):
        break
    else:
        sleep(60*1)

utils.send_line('{} start'.format(__file__))
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
         'learning_rate': 0.1,
         'max_depth': 3,
         'num_leaves': 2**3-1,
         'max_bin': 100,
         'min_child_samples': 300,
         'min_child_weight': 0,
         'colsample_bytree': 0.8,
         'subsample': 0.1,
         'nthread': 64,
         'scale_pos_weight': 500,
         'seed': SEED}

gc.collect()

models = []
for i in range(LOOP):
    gc.collect()
    param.update({'seed':np.random.randint(9999)})
    model = lgb.train(param, dtrain, NROUND)
    model.save_model('lgb{}.model'.format(i))
    models.append(model)
    
del dtrain; gc.collect()


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

