#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:14:03 2018

@author: Kazuki
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 12:21:12 2018

@author: kazuki.onodera
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
NROUND = 300
LOOP = 3
SUBMIT_FILE_PATH = '../output/429-2_drop_ip-device-os-channel.csv.gz'
COMMENT = F"reproduce 429-2 and drop_ip-device-os-channel"

EXE_SUBMIT = True

np.random.seed(SEED)
print('seed :', SEED)

# =============================================================================
# wait
# =============================================================================
while True:
    if os.path.isfile('SUCCESS_803'):
        break
    else:
        sleep(60*1)

utils.send_line(f'START {__file__}')
# =============================================================================
# load train
# =============================================================================
train = utils.read_pickles('../data/dtrain_429-2').drop(['device', 'os', 'channel'], axis=1)
gc.collect()
dtrain = lgb.Dataset(train.drop('is_attributed', axis=1), label=train.is_attributed,
                     categorical_feature=['app', 'hour'])
gc.collect()

del train; gc.collect()

# =============================================================================
# xgboost
# =============================================================================

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.1,
         'max_depth': 4,
         'num_leaves': 2**4-1,
         'max_bin': 100,
         'min_child_samples': 300,
         'min_child_weight': 0,
         'colsample_bytree': 0.8,
         'subsample': 0.1,
         'nthread': 64,
         'scale_pos_weight': 100,
         'seed': SEED}

gc.collect()

models = []
for i in range(LOOP):
    gc.collect()
    param.update({'seed':np.random.randint(9999)})
    model = lgb.train(param, dtrain, NROUND, categorical_feature=['app', 'hour'])
    model.save_model(f'lgb{i}.model')
    models.append(model)
    
del dtrain; gc.collect()
"""

models = []
for i in range(3):
    bst = lgb.Booster(model_file=f'lgb{i}.model')
    models.append(bst)

imp = ex.getImp(models)

"""

# =============================================================================
# test
# =============================================================================

sub = pd.read_pickle('../data/sub_429-2.p')
dtest = utils.read_pickles('../data/dtest_429-2').drop(['device', 'os', 'channel'], axis=1)
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


