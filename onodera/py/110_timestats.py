#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 00:03:49 2018

@author: Kazuki

category:
    ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']

"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import os
from glob import glob
from multiprocessing import Pool
proc = 10
import utils
utils.start(__file__)

# =============================================================================
# load
# =============================================================================
train = pd.concat([utils.read_pickles('../data/train', ['ip', 'app', 'device', 'os', 'channel', 'click_time']),
                   utils.read_pickles('../data/001_train', ['timestamp'])], axis=1)

test = pd.concat([utils.read_pickles('../data/test_old').drop(['click_id', 'click_time'], axis=1),
                   utils.read_pickles('../data/001_test', ['timestamp'])], axis=1)

gc.collect()

trte = pd.concat([train, test])

del train, test; gc.collect()

# =============================================================================
# features
# =============================================================================

def multi(keys):
    keys_ = '-'.join(keys)
    print(keys)
    
    gr = trte.groupby(keys)
    df_min = gr['timestamp'].min()
    df_min.name = 'timemin_' + keys_
    
    df_max = gr['timestamp'].max()
    df_max.name = 'timemax_' + keys_
    
    df_diff1 = df_max - df_min
    df_diff1.name = 'timediff-minmax_' + keys_
    
    df_diff1 = df_diff1.rank(method='dense')
    df_max  = df_max.rank(method='dense')
    df_min  = df_min.rank(method='dense')
    
    gc.collect()
    
    df_mean = gr['timestamp'].mean()
    df_mean.name = 'timemean_' + keys_
    
    df_median = gr['timestamp'].median()
    df_median.name = 'timemedian_' + keys_
    
    df_diff2 = df_mean - df_median
    df_diff2.name = 'timediff-meadian_' + keys_
    
    df_diff2  = df_diff2.rank(method='dense')
    df_median = df_median.rank(method='dense')
    df_mean   = df_mean.rank(method='dense')
    
    gc.collect()
    
    df_var = gr['timestamp'].var().rank(method='dense')
    df_var.name = 'timevar_' + keys_
    
    df_skew = gr['timestamp'].skew().rank(method='dense')
    df_skew.name = 'timeskew_' + keys_
    
    
    df = pd.concat([df_min, df_max, df_diff1, 
                    df_mean, df_median, df_diff2,
                    df_var, df_skew], axis=1)
    del df_min, df_max, df_diff1, df_diff2, df_mean, df_var, df_skew; gc.collect()
    
    utils.reduce_memory(df, ix_start=-1)
    col = df.columns.tolist()
    df.reset_index(inplace=True)
    
    result = pd.merge(trte, df, on=keys, how='left')
    
    result.iloc[0:utils.TRAIN_SHAPE][col].to_pickle('../data/110__{}_train.p'.format(keys))
    result.iloc[utils.TRAIN_SHAPE:][col].to_pickle('../data/110__{}_test.p'.format(keys))
    gc.collect()



# =============================================================================
# concat
# =============================================================================
st = 0
end = 0
limit = 10
for pt in range(1, 10):
    end +=limit
    print(st, end)
    gc.collect()
    pool = Pool(proc)
    callback = pool.map(multi, utils.comb[st:end])
    pool.close()
    st = end
    
    # train
    df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/110__*_train.p'))], axis=1).reset_index(drop=True)
    utils.to_pickles(df, '../data/110-{}_train'.format(pt), 10)
    
    del df; gc.collect()
    
    # test
    df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/110__*_test.p'))], axis=1).reset_index(drop=True)
    utils.to_pickles(df, '../data/110-{}_test'.format(pt), 10)
    
    os.system('rm -rf ../data/110__*.p')
    
    if end >= len(utils.comb):
        break



#==============================================================================
utils.end(__file__)
