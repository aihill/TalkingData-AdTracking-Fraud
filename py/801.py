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


for keys in tqdm(utils.comb):
    gc.collect()
    keys_ = '-'.join(keys)
    df = pd.merge(pd.read_pickle('../data/{}_count_old.p'.format(keys_)),
                  pd.read_pickle('../data/{}_timestd_old.p'.format(keys_)),
                  on=keys, how='outer')
    df.to_pickle('../data/{}_feature.p'.format(keys_))
    