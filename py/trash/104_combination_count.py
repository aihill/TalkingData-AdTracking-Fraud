#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:46:18 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import gc
import utils


comb = list(combinations(['ip', 'app', 'device', 'os', 'channel'], 4))
comb += list(combinations(['ip', 'app', 'device', 'os', 'channel'], 3))
comb += list(combinations(['ip', 'app', 'device', 'os', 'channel'], 2))


# =============================================================================
# for valid
# =============================================================================
valid = utils.read_pickles('../data/valid_feature')


for keys in tqdm(comb):
    keys_ = '-'.join(keys)
    df = valid.groupby(keys).size()
    df.name = keys_+'_count'
    df = df.reset_index()
    df.to_pickle('../data/104_{}_valid.p'.format(keys_))

del valid; gc.collect()


# =============================================================================
# for test
# =============================================================================
test = utils.read_pickles('../data/test_feature')

for keys in tqdm(comb):
    keys_ = '-'.join(keys)
    df = test.groupby(keys).size()
    df.name = keys_+'_count'
    df = df.reset_index()
    df.to_pickle('../data/104_{}_test.p'.format(keys_))


