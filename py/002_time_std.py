#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 02:23:57 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
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

for keys in tqdm(utils.comb):
    keys_ = '-'.join(keys)
    print(keys)
    df = trte.groupby(keys)['timestamp'].std()
    df.name = keys_+'_timestd'
    df = df.reset_index()
    df.to_pickle('../data/{}_timestd_old.p'.format(keys_))







#==============================================================================
utils.end(__file__)
