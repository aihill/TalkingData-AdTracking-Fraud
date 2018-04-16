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
proc = 5
from itertools import combinations
import utils
utils.start(__file__)

os.system('rm -rf ../data/106__*.p')

trte = pd.concat([utils.read_pickles('../data/train'), 
                  utils.read_pickles('../data/test_old')])

#trte['day']  = trte.click_time.dt.day
#trte['hour'] = trte.click_time.dt.hour

gc.collect()



def multi(keys):
    gc.collect()
    print(keys)
    keys1, keys2 = keys
    
    df = trte.groupby(keys1).size().groupby(keys2).size()
    c = 'nunique_' + '-'.join(keys1) + '_' + '-'.join(keys2)
    df.name = c
    df = df.reset_index()
    
    result = pd.merge(trte, df, on=keys2, how='left')
    
    result.iloc[0:utils.TRAIN_SHAPE][c].to_pickle('../data/106__{}_train.p'.format(c))
    result.iloc[utils.TRAIN_SHAPE:][c].to_pickle('../data/106__{}_test.p'.format(c))
    gc.collect()


comb = []
tmp1 = list(combinations(['ip', 'app', 'device', 'os', 'channel'], 4))
for c1 in tmp1:
    tmp2 = list(combinations(c1, 2))
    for c2 in tmp2:
        comb.append( [list(c1), list(c2)] )


# =============================================================================
# concat pt1
# =============================================================================
gc.collect()
pool = Pool(proc)
callback = pool.map(multi, comb[:10])
pool.close()

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/106__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/106-1_train', 10)

gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/106__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/106-1_test', 10)

os.system('rm -rf ../data/106__*.p')



# =============================================================================
# concat pt2
# =============================================================================
gc.collect()
pool = Pool(proc)
callback = pool.map(multi, comb[10:20])
pool.close()

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/106__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/106-2_train', 10)

gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/106__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/106-2_test', 10)

os.system('rm -rf ../data/106__*.p')



# =============================================================================
# concat pt3
# =============================================================================
gc.collect()
pool = Pool(proc)
callback = pool.map(multi, comb[20:])
pool.close()

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/106__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/106-3_train', 10)

gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/106__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/106-3_test', 10)

os.system('rm -rf ../data/106__*.p')




#==============================================================================
utils.end(__file__)


