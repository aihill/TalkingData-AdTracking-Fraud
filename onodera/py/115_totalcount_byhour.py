#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 11:12:26 2018

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

os.system('rm -rf ../data/115__*.p')

trte = pd.concat([utils.read_pickles('../data/train'),
                utils.read_pickles('../data/test_old')])
trte['hour']  = (trte.click_time + pd.offsets.Hour(8)).dt.hour

day_tbl = trte['hour'].value_counts().reset_index()
day_tbl.columns = ['hour', 'hour_freq']

def multi(keys):
    gc.collect()
    print(keys)
    keys = list(keys)
    
    keys_ = '-'.join(keys)
    c1 = 'totalCountByHour_' + keys_
    c2 = 'totalRatioByHour_' + keys_
    
    keys +=['hour']
    df = trte.groupby(keys).size()
    df.name = c1
    df = pd.merge(df.reset_index(), day_tbl, on='hour', how='left')
    df[c2] = df[c1] / df['hour_freq']
    del df['hour_freq']
    
    utils.reduce_memory(df, ix_start=-2)
    
    result = pd.merge(trte, df, on=keys, how='left')
    
    result.iloc[0:utils.TRAIN_SHAPE][[c1, c2]].to_pickle('../data/115__{}_train.p'.format(keys_))
    result.iloc[utils.TRAIN_SHAPE:][[c1, c2]].to_pickle('../data/115__{}_test.p'.format(keys_))
    gc.collect()
    
pool = Pool(10)
callback = pool.map(multi, utils.comb)
pool.close()

# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/115__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/115_train', utils.SPLIT_SIZE)

gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/115__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/115_test', utils.SPLIT_SIZE)

os.system('rm -rf ../data/115__*.p')





#==============================================================================
utils.end(__file__)


