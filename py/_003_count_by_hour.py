#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 17:54:54 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import utils
utils.start(__file__)


trte = pd.concat([utils.read_pickles('../data/train'),
                utils.read_pickles('../data/test_old')])

trte['hour'] = trte.click_time.dt.hour

for keys in tqdm(utils.comb):
    keys_ = '-'.join(keys)
    df = trte.groupby(keys+['hour']).size()
    df.name = keys_+'_count_byhour'
    df = df.reset_index()
    df.to_pickle('../data/{}_count_byhour_old.p'.format(keys_))

#==============================================================================
utils.end(__file__)

