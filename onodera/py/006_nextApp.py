#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 20:47:56 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
from time import time
import gc
import os
from glob import glob
from multiprocessing import Pool
nthread = 5
from itertools import combinations
import utils
utils.start(__file__)


os.system('rm -rf ../data/006__*.p')

trte = pd.concat([utils.read_pickles('../data/train', ['ip', 'app', 'device', 'os', 'channel', 'click_time']),
                utils.read_pickles('../data/test_old', ['ip', 'app', 'device', 'os', 'channel', 'click_time'])],
                ignore_index=True)

gc.collect()


def multi(count_keys):
    """
    ex:
    count_keys = ('ip', 'app', 'device')
    
    """
    
    if 'app' in count_keys and 'channel' in count_keys:
        return
    
    st = time()
    gc.collect()
    print(count_keys)
    count_keys = list(count_keys)
    
    count_keys_ = '-'.join(count_keys)
    
    df = trte.sort_values(count_keys+['click_time'])
    df['prekey_match']  = ( df[count_keys]==df[count_keys].shift() ).all(1)*1
    df['nextkey_match'] = ( df[count_keys]==df[count_keys].shift(-1) ).all(1)*1
    gc.collect()
    
    col = []
    if 'app' not in count_keys:
        c = 'preApp_'+count_keys_
        col.append(c)
        df[c] = df.app.shift()
        df.loc[df.prekey_match==0, c] = -1
        
        c = 'nextApp_'+count_keys_
        col.append(c)
        df[c] = df.app.shift(-1)
        df.loc[df.nextkey_match==0, c] = -1
        gc.collect()
    
    if 'channel' not in count_keys:
        c = 'preChannel_'+count_keys_
        col.append(c)
        df[c] = df.channel.shift()
        df.loc[df.prekey_match==0, c] = -1
        
        c = 'nextChannel_'+count_keys_
        col.append(c)
        df[c] = df.channel.shift(-1)
        df.loc[df.nextkey_match==0, c] = -1
        gc.collect()
    
    
    
    df.drop(count_keys, axis=1, inplace=True)
    df.sort_index(inplace=True)
    df.fillna(-1, inplace=True)
    gc.collect()
    
    df.iloc[0:utils.TRAIN_SHAPE][col].to_pickle(f'../data/005__{count_keys_}_train.p')
    df.iloc[utils.TRAIN_SHAPE:][col].to_pickle(f'../data/005__{count_keys_}_test.p')
    
    print(f'Finished {count_keys} {(time()-st)/60:.3f} min')
    

pool = Pool(nthread)
callback = pool.map(multi, utils.comb)
pool.close()

del trte; gc.collect()

# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/006__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/006_train', 10)

print(df.columns.tolist())
del df; gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/006__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/006_test', 10)

os.system('rm -rf ../data/006__*.p')



#==============================================================================
utils.end(__file__)

