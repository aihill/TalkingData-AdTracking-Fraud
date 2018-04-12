#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 23:03:51 2018

@author: kazuki.onodera
"""

import os
import pandas as pd
#from tqdm import tqdm
import gc
from glob import glob
from multiprocessing import Pool
import utils
utils.start(__file__)

# =============================================================================
# load
# =============================================================================

train = pd.concat([utils.read_pickles('../data/train', ['ip', 'app', 'device', 'os', 'channel']),
                   utils.read_pickles('../data/004_train', ['timestamp'])], axis=1)

test = pd.concat([utils.read_pickles('../data/test_old', ['ip', 'app', 'device', 'os', 'channel']),
                  utils.read_pickles('../data/004_test', ['timestamp'])], axis=1)

gc.collect()

trte = pd.concat([train, test])

del train, test; gc.collect()

# =============================================================================
# features
# =============================================================================

# std
def multi(keys):
    keys_ = '-'.join(keys)
    print(keys)
    
    df_min = trte.groupby(keys)['timestamp'].min()
    df_min.name = 'timemin_' + keys_ 
    
    df_max = trte.groupby(keys)['timestamp'].max()
    df_max.name = 'timemax_' + keys_
    
    df_mean = trte.groupby(keys)['timestamp'].mean()
    df_mean.name = 'timemean_' + keys_
    
    df_std = trte.groupby(keys)['timestamp'].std()
    df_std.name = 'timestd_' + keys_
    
    df = pd.concat([df_min, df_max, df_mean, df_std], axis=1).reset_index()
    df['timemax-min_'+keys_] = df['timemax_' + keys_] - df['timemin_' + keys_]
    
    col = [c for c in df.columns if c.startswith('time')]
    result = pd.merge(trte, df, on=keys, how='left')
    result.iloc[0:184903890][col].to_pickle('../data/103__{}_train.p'.format(keys_))
    result.iloc[184903890:][col].to_pickle('../data/103__{}_test.p'.format(keys_))
    
    gc.collect()


pool = Pool(10)
callback = pool.map(multi, utils.comb)
pool.close()

# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/103__*_train.p'))], axis=1)
utils.to_pickles(df, '../data/103_train', 10)

gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/103__*_test.p'))], axis=1)
utils.to_pickles(df, '../data/103_test', 10)

os.system('rm -rf ../data/103__*.p')



#==============================================================================
utils.end(__file__)
