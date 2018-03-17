#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:02:42 2018

@author: kazuki.onodera
"""

import pandas as pd
import sys
sys.path.append('/home/kazuki_onodera/Python')
import xgbextension as ex
import xgboost as xgb
import gc
import utils

# setting
submit_file_path = '../output/315-1.csv.gz'

# =============================================================================
# load valid
# =============================================================================

valid = pd.concat([utils.read_pickles('../data/valid').sort_values(utils.sort_keys).reset_index(drop=True),
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

gc.collect()

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

y = valid.is_attributed
print(valid.columns.tolist())
valid.drop(['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed'], 
           axis=1, inplace=True)

valid.fillna(-1, inplace=True)

print('valid.shape:', valid.shape)
valid_head = valid.head()
valid_head.to_pickle('valid_head.p')

dtrain = xgb.DMatrix(valid, y)
del valid, y; gc.collect()

print('start xgb')
model = xgb.train(param, dtrain, 1000)
model.save_model('xgb.model')

imp = ex.getImp(model)
imp.to_csv('imp.csv', index=False)

del dtrain; gc.collect()

# =============================================================================
# test
# =============================================================================
test = pd.concat([utils.read_pickles('../data/test').sort_values(utils.sort_keys).reset_index(drop=True),
                   pd.read_pickle('../data/101_test.p').reset_index(drop=True),
                   pd.read_pickle('../data/103_test.p').reset_index(drop=True)],
                  axis=1)

gc.collect()
test = pd.merge(test, pd.read_pickle('../data/102_app_test.p'),
                 on='app', how='left')
test = pd.merge(test, pd.read_pickle('../data/app.p'),
                 on='app', how='left')

gc.collect()
test = pd.merge(test, pd.read_pickle('../data/102_channel_test.p'),
                 on='channel', how='left')
test = pd.merge(test, pd.read_pickle('../data/channel.p'),
                 on='channel', how='left')

gc.collect()
test = pd.merge(test, pd.read_pickle('../data/102_device_test.p'),
                 on='device', how='left')
test = pd.merge(test, pd.read_pickle('../data/device.p'),
                 on='device', how='left')

gc.collect()
test = pd.merge(test, pd.read_pickle('../data/102_ip_test.p'),
                 on='ip', how='left')
test = pd.merge(test, pd.read_pickle('../data/ip.p'),
                 on='ip', how='left')

gc.collect()
test = pd.merge(test, pd.read_pickle('../data/102_os_test.p'),
                 on='os', how='left')
test = pd.merge(test, pd.read_pickle('../data/os.p'),
                 on='os', how='left')

gc.collect()

test.fillna(-1, inplace=True)

dtest = xgb.DMatrix(test[valid_head.columns])
y_pred = model.predict(dtest)

sub = test[['click_id']]
sub['is_attributed'] = y_pred

sub.to_csv(submit_file_path, index=False, compression='gzip')

# =============================================================================
# submission
# =============================================================================
utils.submit(submit_file_path)







