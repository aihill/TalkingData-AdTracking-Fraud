#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 02:23:57 2018

@author: Kazuki

takes 90 minutes

"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from multiprocessing import Pool
import utils
utils.start(__file__)

# =============================================================================
# load
# =============================================================================
col_drop = ['click_time', 'attributed_time', 'is_attributed']
train = pd.concat([utils.read_pickles('../data/train').drop(col_drop, axis=1),
                   pd.read_pickle('../data/102_train.p').drop('hour', axis=1)], axis=1)

test = pd.concat([utils.read_pickles('../data/test_old').drop(['click_id', 'click_time'], axis=1),
                   pd.read_pickle('../data/102_test_old.p').drop('hour', axis=1)], axis=1)

gc.collect()

trte = pd.concat([train, test])

del train, test; gc.collect()

# =============================================================================
# features
# =============================================================================

# std
def multi(keys):
    keys_ = '-'.join(keys)
    print(keys)
    
    df_min = trte.groupby(keys)['timestamp'].min()
    df_min.name = keys_+'_timemin'
    
    df_max = trte.groupby(keys)['timestamp'].max()
    df_max.name = keys_+'_timemax'
    
    df_mean = trte.groupby(keys)['timestamp'].mean()
    df_mean.name = keys_+'_timemean'
    
    df_std = trte.groupby(keys)['timestamp'].std()
    df_std.name = keys_+'_timestd'
    
    
    df = pd.concat([df_min, df_max, df_mean, df_std], axis=1).reset_index()
    df.to_pickle('../data/{}_timestats_old.p'.format(keys_))



pool = Pool(32)
callback = pool.map(multi, utils.comb)
pool.close()


#==============================================================================
utils.end(__file__)
