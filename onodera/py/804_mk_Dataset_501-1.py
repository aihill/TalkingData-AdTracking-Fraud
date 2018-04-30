#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 23:58:16 2018

@author: Kazuki
"""

from glob import glob
import pandas as pd
from os import system
import os
from time import sleep
from tqdm import tqdm
import gc
import lightgbm as lgb
from multiprocessing import Pool
import utils
utils.start(__file__)

system('rm SUCCESS_804')

categorical_feature = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']

# =============================================================================
# wait
# =============================================================================
while True:
    if os.path.isfile('SUCCESS_803'):
        break
    else:
        sleep(60*1)

utils.send_line('START {}'.format(__file__))

# =============================================================================
# train
# =============================================================================
train = utils.read_pickles('../data/dtrain')

for c in categorical_feature:
    col = ['is_attributed', c]
    filepath = f'../data/dtrain_drop_{c}.mt'
    
    categorical_feature_ = list( set(categorical_feature) - set([c]) )
    
    print(f'writing {filepath}...')
    system(f'rm {filepath}')
    lgb.Dataset(train.drop(col, axis=1), label=train.is_attributed,
                categorical_feature=categorical_feature_
                ).save_binary(filepath)
    
    gc.collect()


del train; gc.collect()

# =============================================================================
# test
# =============================================================================
test = utils.read_pickles('../data/dtest')
X_head = pd.read_pickle('X_head.p')

for c in categorical_feature:
    col = c
    filepath = f'../data/dtest_drop_{c}'
    
    categorical_feature_ = list( set(categorical_feature) - set([c]) )
    
    print(f'writing {filepath}...')
    system(f'rm -rf {filepath}')
    utils.to_pickles(test[X_head.columns], filepath, utils.SPLIT_SIZE)
    
    gc.collect()


del test; gc.collect()




#==============================================================================
system('touch SUCCESS_804')

utils.end(__file__)

