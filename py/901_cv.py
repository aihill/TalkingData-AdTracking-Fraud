#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:04:05 2018

@author: Kazuki

sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"

"""

import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('/home/kazuki_onodera/Python')
import xgbextension as ex
import xgboost as xgb
import gc
import utils

# =============================================================================
# load valid
# =============================================================================

valid = pd.concat([utils.read_pickles('../data/valid').sort_values(utils.sort_keys).drop(['click_time', 'attributed_time'], axis=1).reset_index(drop=True),
                   pd.read_pickle('../data/101_valid.p').reset_index(drop=True),
                   pd.read_pickle('../data/103_valid.p').reset_index(drop=True)],
                  axis=1)

gc.collect()
valid = pd.merge(valid, pd.read_pickle('../data/102_app_valid.p'),
                 on='app', how='left')
valid = pd.merge(valid, pd.read_pickle('../data/app.p'),
                 on='app', how='left')

gc.collect()
valid = pd.merge(valid, pd.read_pickle('../data/102_channel_valid.p'),
                 on='channel', how='left')
valid = pd.merge(valid, pd.read_pickle('../data/channel.p'),
                 on='channel', how='left')

gc.collect()
valid = pd.merge(valid, pd.read_pickle('../data/102_device_valid.p'),
                 on='device', how='left')
valid = pd.merge(valid, pd.read_pickle('../data/device.p'),
                 on='device', how='left')

gc.collect()
valid = pd.merge(valid, pd.read_pickle('../data/102_ip_valid.p'),
                 on='ip', how='left')
valid = pd.merge(valid, pd.read_pickle('../data/ip.p'),
                 on='ip', how='left')

gc.collect()
valid = pd.merge(valid, pd.read_pickle('../data/102_os_valid.p'),
                 on='os', how='left')
valid = pd.merge(valid, pd.read_pickle('../data/os.p'),
                 on='os', how='left')

# 104
from itertools import combinations
comb = list(combinations(['ip', 'app', 'device', 'os', 'channel'], 4))
comb += list(combinations(['ip', 'app', 'device', 'os', 'channel'], 3))
comb += list(combinations(['ip', 'app', 'device', 'os', 'channel'], 2))

for keys in tqdm(comb):
    gc.collect()
    keys_ = '-'.join(keys)
    df = pd.read_pickle('../data/104_{}_valid.p'.format(keys_))
    valid = pd.merge(valid, df, on=keys, how='left')



gc.collect()

param = {'colsample_bylebel': 0.8,
         'subsample': 0.5,
         'eta': 0.1,
         'eval_metric': 'auc',
         'max_depth': 6,
         'objective': 'binary:logistic',
         'silent': 1,
         'tree_method': 'hist',
         'nthread': 64,
         'seed':71}

y = valid.is_attributed
print(valid.columns.tolist())
valid.drop(['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 
           axis=1, inplace=True)

valid.fillna(-1, inplace=True)

yhat, imp, ret = ex.stacking(valid, y, param, 9999, nfold=5, esr=30)

imp.to_csv('imp.csv', index=False)

