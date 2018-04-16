#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:10:18 2018

@author: kazuki.onodera

category:
    ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']


"""

import os
import pandas as pd
from tqdm import tqdm
import gc
from glob import glob
from multiprocessing import Pool
import utils
utils.start(__file__)



trte = pd.concat([pd.concat([utils.read_pickles('../data/001_train'),
                             utils.read_pickles('../data/001_test')]), 
                  pd.concat([utils.read_pickles('../data/train'), 
                             utils.read_pickles('../data/test_old')])], 
                 axis=1)
gc.collect()



def multi(keys):
    gc.collect()
    print(keys)
    keys1, keys2 = keys
    
    df = trte.groupby(keys1).size().reset_index().groupby(keys2).size().reset_index()
    c = 'nunique_' + '-'.join(keys1) + '_' + '-'.join(keys2)
    df.name = c
    df = df.reset_index()
    
    result = pd.merge(trte, df, on=keys2, how='left')
    
    result.iloc[0:utils.TRAIN_SHAPE][c].to_pickle('../data/102__{}_train.p'.format(c))
    result.iloc[utils.TRAIN_SHAPE:][c].to_pickle('../data/102__{}_test.p'.format(c))
    gc.collect()


comb = [
        [['ip', 'app'],     ['ip']],
        [['ip', 'device'],  ['ip']],
        [['ip', 'os'],      ['ip']],
        [['ip', 'channel'], ['ip']],
        [['ip', 'day'],     ['ip']],
        [['ip', 'hour'],    ['ip']],
        
        [['ip', 'app'],     ['app']],
        [['ip', 'device'],  ['device']],
        [['ip', 'os'],      ['os']],
        [['ip', 'channel'], ['channel']],
        [['ip', 'day'],     ['day']],
        [['ip', 'hour'],    ['hour']],
        
        [['ip', 'os', 'device'], ['ip']],
        [['ip', 'os', 'device'], ['ip', 'os']],
        [['ip', 'os', 'device'], ['ip', 'device']],
        [['ip', 'os', 'device'], ['os']],
        [['ip', 'os', 'device'], ['device']],
        
        [['ip', 'app', 'device', 'os'],['']],
        
        [['ip', 'day', 'hour'], ['ip', 'day']],
        
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


