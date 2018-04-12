#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:10:18 2018

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
    keys1, keys2 = keys
    
    keys1_ = '-'.join(keys1)
    df = trte.groupby(keys1).size()
    
    keys2_ = '-'.join(keys2)
    df = df.reset_index().groupby(keys2).size()
    c = 'totalcount2_' + keys2_ + '_' + keys1_
    df.name = c
    result = pd.merge(trte, df, on=keys2, how='left')
    
    result.iloc[0:184903890][c].to_pickle('../data/102__{}_train.p'.format(c))
    result.iloc[184903890:][c].to_pickle('../data/102__{}_test.p'.format(c))
    gc.collect()


comb = [
        [['ip', 'os', 'device'], ['ip']],
        [['ip', 'os', 'device'], ['ip', 'os']],
        
        ]
pool = Pool(10)
callback = pool.map(multi, comb)
pool.close()

# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/102__*_train.p'))], axis=1)
utils.to_pickles(df, '../data/102_train', 10)

gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/102__*_test.p'))], axis=1)
utils.to_pickles(df, '../data/102_test', 10)

os.system('rm -rf ../data/102__*.p')





#==============================================================================
utils.end(__file__)


