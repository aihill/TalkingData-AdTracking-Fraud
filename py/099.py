#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:21:44 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool
from tqdm import tqdm
import utils
utils.start(__file__)

train = utils.read_pickles('../data/train')
test  = utils.read_pickles('../data/test_old')

def multi(keys):
    gc.collect()
    keys_ = '-'.join(keys)
    df = pd.merge(pd.read_pickle('../data/{}_count_old.p'.format(keys_)),
                  pd.read_pickle('../data/{}_timestats_old.p'.format(keys_)),
                  on=keys, how='outer')
    df = pd.merge(df, pd.read_pickle('../data/{}_sametimestats_old.p'.format(keys_)),
                  on=keys, how='outer')
    utils.reduce_memory(df)
    
    col = [c for c in df.columns if c not in keys]
    train_ = pd.merge(train, df, on=keys, how='left')
    train_[col].to_pickle('../data/{}_train.p'.format(keys_))
    train_[col].tail(999999).to_pickle('../data/{}_train_tail.p'.format(keys_))
    
    test_ = pd.merge(test, df, on=keys, how='left')
    test_[col].to_pickle('../data/{}_test.p'.format(keys_)) # suffix should be 'old'?
    test_[col].tail(999999).to_pickle('../data/{}_test_tail.p'.format(keys_)) # suffix should be 'old'?
    
pool = Pool(6)
callback = pool.map(multi, utils.comb)
pool.close()

#==============================================================================
utils.end(__file__)
