#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 12:39:39 2018

@author: Kazuki
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
comb += list(combinations(['ip', 'app', 'device', 'os', 'channel'], 1))


# =============================================================================
# for valid
# =============================================================================
trte = pd.concat([utils.read_pickles('../data/train'),
                utils.read_pickles('../data/test')])


for keys in tqdm(comb):
    keys_ = '-'.join(keys)
    df = trte.groupby(keys).size()
    df.name = keys_+'_count'
    df = df.reset_index()
    df.to_pickle('../data/{}_count.p'.format(keys_))

