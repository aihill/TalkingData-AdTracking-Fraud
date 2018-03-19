#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 18:28:14 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import utils
utils.start(__file__)

# setting
SEED = 48

np.random.seed(SEED)

# =============================================================================
# load train & feature
# =============================================================================

train = pd.concat([utils.read_pickles('../data/train'),
                   pd.read_pickle('../data/101_train.p'),
                   pd.read_pickle('../data/102_train.p')], axis=1)#.sample(frac=0.4)

gc.collect()

for keys in tqdm(utils.comb):
    gc.collect()
    keys_ = '-'.join(keys)
    train = pd.merge(train, pd.read_pickle('../data/{}_feature.p'.format(keys_)), 
                     on=keys, how='left')

train.drop(['click_time', 'attributed_time'], axis=1, inplace=True)

y = train.is_attributed
train.drop(['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 
           axis=1, inplace=True)
train.fillna(-1, inplace=True)

# =============================================================================
# 
# =============================================================================





