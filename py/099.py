#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:21:44 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
import utils
utils.start(__file__)

train = utils.read_pickles('../data/train')
test  = utils.read_pickles('../data/test_old')

for keys in tqdm(utils.comb):
    gc.collect()
    keys_ = '-'.join(keys)
    df = pd.merge(pd.read_pickle('../data/{}_count_old.p'.format(keys_)),
                  pd.read_pickle('../data/{}_timestd_old.p'.format(keys_)),
                  on=keys, how='outer')
    utils.reduce_memory(df)
    
    col = [c for c in df.columns if c not in keys]
    train_ = pd.merge(train, df, on=keys, how='left')
    train_[col].drop(keys, axis=1).to_pickle('../data/{}_train.p'.format(keys_))
    test_ = pd.merge(test, df, on=keys, how='left')
    test_[col].drop(keys, axis=1).to_pickle('../data/{}_test.p'.format(keys_))
    
#==============================================================================
utils.end(__file__)
