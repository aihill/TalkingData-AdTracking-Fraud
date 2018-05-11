#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 00:48:56 2018

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

os.system('rm -rf ../data/008*')

# =============================================================================
# train
# =============================================================================
train = utils.read_pickles('../data/train', ['ip', 'app', 'device', 'os', 'channel', 'click_time'])

train['day']  = (train.click_time + pd.offsets.Hour(8)).dt.day


def multi_train(count_keys):
    """
    ex:
    count_keys = ('app', 'device')
    
    """
    
    gc.collect()
    print(count_keys)
    count_keys = list(count_keys)
    
    count_keys_ = '-'.join(count_keys)
    keys = count_keys+['click_time']
    df = train[keys].sort_values(keys)
    
    gc.collect()
    c1 = 'timedelta_'+count_keys_
    df[c1] = df.click_time.diff().dt.seconds
    df['key_match'] = ( df[count_keys]==df[count_keys].shift() ).all(1)*1
    df.loc[df.key_match==0, c1] = -1
    
    gc.collect()
    c2 = 'timedelta_rev_'+count_keys_
    df[c2] = df.click_time.diff(-1).dt.seconds.abs()
    df['key_match'] = ( df[count_keys]==df[count_keys].shift(-1) ).all(1)*1
    df.loc[df.key_match==0, c2] = -1
    
    df.drop(count_keys, axis=1, inplace=True)
    df.sort_index(inplace=True)
    
    gc.collect()
    df[[c1, c2]].to_pickle('../data/008__{}_train.p'.format(count_keys_))


pool = Pool(nthread)
callback = pool.map(multi_train, utils.comb)
pool.close()

# =============================================================================
# test
# =============================================================================
test = utils.read_pickles('../data/test_old', ['ip', 'app', 'device', 'os', 'channel', 'click_time'])

def multi_test(count_keys):
    """
    ex:
    count_keys = ('app', 'device')
    
    """
    
    gc.collect()
    print(count_keys)
    count_keys = list(count_keys)
    
    count_keys_ = '-'.join(count_keys)
    keys = count_keys+['click_time']
    df = test[keys].sort_values(keys)
    
    gc.collect()
    c1 = 'timedelta_'+count_keys_
    df[c1] = df.click_time.diff().dt.seconds
    df['key_match'] = ( df[count_keys]==df[count_keys].shift() ).all(1)*1
    df.loc[df.key_match==0, c1] = -1
    
    gc.collect()
    c2 = 'timedelta_rev_'+count_keys_
    df[c2] = df.click_time.diff(-1).dt.seconds.abs()
    df['key_match'] = ( df[count_keys]==df[count_keys].shift(-1) ).all(1)*1
    df.loc[df.key_match==0, c2] = -1
    
    df.drop(count_keys, axis=1, inplace=True)
    df.sort_index(inplace=True)
    
    gc.collect()
    df[[c1, c2]].to_pickle('../data/008__{}_test.p'.format(count_keys_))


pool = Pool(nthread)
callback = pool.map(multi_test, utils.comb)
pool.close()


# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/008__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/008_train', utils.SPLIT_SIZE)

del df; gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/008__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/008_test', utils.SPLIT_SIZE)

os.system('rm -rf ../data/008__*.p')


#==============================================================================
utils.end(__file__)

