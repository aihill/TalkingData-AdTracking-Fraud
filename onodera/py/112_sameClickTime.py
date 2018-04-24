#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 03:07:21 2018

@author: Kazuki
"""


import os
import pandas as pd
from tqdm import tqdm
import gc
from glob import glob
from multiprocessing import Pool
import utils
utils.start(__file__)

os.system('rm -rf ../data/112__*.p')

trte = pd.concat([utils.read_pickles('../data/train',    ['ip', 'app', 'device', 'os', 'channel', 'click_time']),
                  utils.read_pickles('../data/test_old', ['ip', 'app', 'device', 'os', 'channel', 'click_time'])])

def multi(keys):
    gc.collect()
    print(keys)
    
    keys_ = '-'.join(keys)
    c = 'sameClickTimeCount_' + keys_
    df = trte.groupby(keys+['click_time']).size().groupby(keys).max().rank(method='dense')
    df.name = c
    df = df.reset_index()
    utils.reduce_memory(df, ix_start=-1)
    gc.collect()
    
    result = pd.merge(trte, df, on=keys, how='left')
    gc.collect()
    
    result.iloc[0:utils.TRAIN_SHAPE][c].to_pickle('../data/112__{}_train.p'.format(keys_))
    result.iloc[utils.TRAIN_SHAPE:][c].to_pickle('../data/112__{}_test.p'.format(keys_))
    gc.collect()
    
pool = Pool(5)
callback = pool.map(multi, utils.comb)
pool.close()

# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/112__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/112_train', utils.SPLIT_SIZE)

gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/112__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/112_test', utils.SPLIT_SIZE)

os.system('rm -rf ../data/112__*.p')





#==============================================================================
utils.end(__file__)


