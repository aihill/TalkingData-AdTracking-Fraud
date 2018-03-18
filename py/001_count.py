#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 12:39:39 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import utils


trte = pd.concat([utils.read_pickles('../data/train'),
                utils.read_pickles('../data/test_old')])


for keys in tqdm(utils.comb):
    keys_ = '-'.join(keys)
    df = trte.groupby(keys).size()
    df.name = keys_+'_count'
    df = df.reset_index()
    df.to_pickle('../data/{}_count_old.p'.format(keys_))

