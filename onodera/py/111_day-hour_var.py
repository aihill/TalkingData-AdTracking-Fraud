#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:45:43 2018

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

os.system('rm -rf ../data/111__*.p')

trte = pd.concat([utils.read_pickles('../data/train'), 
                  utils.read_pickles('../data/test_old')])

trte['day']  = trte.click_time.dt.day
trte['hour'] = trte.click_time.dt.hour

gc.collect()

def multi(keys):
    keys_ = '-'.join(keys)
    print(keys)
    
    gr = trte.groupby(keys)
    
    df1 = gr['hour'].var().rank(method='dense')
    c = 'hourvar_' + keys_
    df1.name = c
    
    
    df2 = gr['day'].var().rank(method='dense')
    c = 'dayvar_' + keys_
    df2.name = c
    
    df = pd.concat([df1, df2], axis=1)
    del df1, df2; gc.collect()
    
    utils.reduce_memory(df)
    col = df.columns.tolist()
    df.reset_index(inplace=True)
    
    result = pd.merge(trte, df, on=keys, how='left')
    
    result.iloc[0:utils.TRAIN_SHAPE][col].to_pickle('../data/111__{}_train.p'.format(keys_))
    result.iloc[utils.TRAIN_SHAPE:][col].to_pickle('../data/111__{}_test.p'.format(keys_))
    gc.collect()
    

# =============================================================================
# concat pt1
# =============================================================================
gc.collect()
pool = Pool(proc)
callback = pool.map(multi, utils.comb[:10])
pool.close()

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/111__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/111-1_train', 10)

gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/111__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/111-1_test', 10)

os.system('rm -rf ../data/111__*.p')



# =============================================================================
# concat pt2
# =============================================================================
gc.collect()
pool = Pool(proc)
callback = pool.map(multi, utils.comb[10:20])
pool.close()

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/111__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/111-2_train', 10)

gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/111__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/111-2_test', 10)

os.system('rm -rf ../data/111__*.p')




# =============================================================================
# concat pt3
# =============================================================================
gc.collect()
pool = Pool(proc)
callback = pool.map(multi, utils.comb[20:])
pool.close()

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/111__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/111-3_train', 10)

gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/111__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/111-3_test', 10)

os.system('rm -rf ../data/111__*.p')


#==============================================================================
utils.end(__file__)

