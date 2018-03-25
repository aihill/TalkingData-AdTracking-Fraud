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
    
    c = keys_+'_sametime_count'
    df = trte.groupby(list(keys)+['timestamp']).size()
    df.name = c
    df = df.reset_index()
    
    df_min = df.groupby(keys)[c].min()
    df_min.name = keys_+'_min'
    
    df_max = df.groupby(keys)[c].max()
    df_max.name = keys_+'_max'
    
    df_mean = df.groupby(keys)[c].mean()
    df_mean.name = keys_+'_mean'
    
    df_std = df.groupby(keys)[c].std()
    df_std.name = keys_+'_std'
    
    df = pd.concat([df_min, df_max, df_mean, df_std], axis=1).reset_index()
    df.to_pickle('../data/{}_sametimestats_old.p'.format(keys_))



pool = Pool(16)
callback = pool.map(multi, utils.comb)
pool.close()


#==============================================================================
utils.end(__file__)
