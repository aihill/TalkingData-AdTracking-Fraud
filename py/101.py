#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:08:23 2018

@author: kazuki.onodera
"""

import os
import pandas as pd
from tqdm import tqdm
import gc
from glob import glob
from multiprocessing import Pool
import utils
utils.start(__file__)


trte = pd.concat([utils.read_pickles('../data/train'),
                utils.read_pickles('../data/test_old')])

def multi(keys):
    gc.collect()
    print(keys)
    
    keys_ = '-'.join(keys)
    df = trte.groupby(keys).size()
    df.name = 'totalcount_' + keys_
    df = df.reset_index()
    result = pd.merge(trte, df, on=keys, how='left')
    
    result.iloc[0:utils.TRAIN_SHAPE]['totalcount_' + keys_].to_pickle('../data/101__{}_train.p'.format(keys_))
    result.iloc[utils.TRAIN_SHAPE:]['totalcount_' + keys_].to_pickle('../data/101__{}_test.p'.format(keys_))
    gc.collect()
    
pool = Pool(10)
callback = pool.map(multi, utils.comb)
pool.close()

# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/101__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/101_train', 10)

gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/101__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/101_test', 10)

os.system('rm -rf ../data/101__*.p')





#==============================================================================
utils.end(__file__)


