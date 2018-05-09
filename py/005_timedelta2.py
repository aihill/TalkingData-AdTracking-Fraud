#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 19:30:44 2018

@author: Kazuki
"""


#import numpy as np
import pandas as pd
#from tqdm import tqdm
import gc
import os
from glob import glob
from multiprocessing import Pool
nthread = 12
#from collections import defaultdict
import utils
utils.start(__file__)

os.system('rm -rf ../data/005*')

trte = pd.concat([utils.read_pickles('../data/train', ['ip', 'app', 'device', 'os', 'channel', 'click_time']),
                utils.read_pickles('../data/test_old', ['ip', 'app', 'device', 'os', 'channel', 'click_time'])],
                ignore_index=True)

def multi(count_keys):
    """
    ex:
    count_keys = ('app', 'device')
    
    """
    
    gc.collect()
    print(count_keys)
    count_keys = list(count_keys)
    
    count_keys_ = '-'.join(count_keys)
    keys = count_keys+['click_time']
    df = trte[keys].sort_values(keys)
    
    gc.collect()
    c1 = 'timedelta2_'+count_keys_
    df[c1] = df.click_time.diff(2).dt.seconds
    df['key_match'] = ( df[count_keys]==df[count_keys].shift(2) ).all(1)*1
    df.loc[df.key_match==0, c1] = -1
    
    gc.collect()
    c2 = 'timedelta2_rev_'+count_keys_
    df[c2] = df.click_time.diff(-2).dt.seconds.abs()
    df['key_match'] = ( df[count_keys]==df[count_keys].shift(-2) ).all(1)*1
    df.loc[df.key_match==0, c2] = -1
    
    df.drop(count_keys, axis=1, inplace=True)
    df.sort_index(inplace=True)
    
    gc.collect()
    df.iloc[0:utils.TRAIN_SHAPE][[c1, c2]].to_pickle('../data/005__{}_train.p'.format(count_keys_))
    df.iloc[utils.TRAIN_SHAPE:][[c1, c2]].to_pickle('../data/005__{}_test.p'.format(count_keys_))
    
    print(f'fin {count_keys}')


pool = Pool(nthread)
callback = pool.map(multi, utils.comb)
pool.close()


# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/005__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/005_train', utils.SPLIT_SIZE)

del df; gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/005__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/005_test', utils.SPLIT_SIZE)

os.system('rm -rf ../data/005__*.p')



#==============================================================================
utils.end(__file__)

