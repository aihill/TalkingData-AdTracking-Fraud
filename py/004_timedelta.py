#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:43:36 2018

@author: kazuki.onodera
"""


#import numpy as np
import pandas as pd
#from tqdm import tqdm
import gc
import os
from glob import glob
from multiprocessing import Pool
nthread = 16
#from collections import defaultdict
import utils
utils.start(__file__)

os.system('rm -rf ../data/004*')

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
    
    c1 = 'timedelta_'+count_keys_
    df[c1] = df.click_time.diff().dt.seconds
    df['key_match'] = ( df[count_keys]==df[count_keys].shift() ).all(1)*1
    df.loc[df.key_match==0, c1] = -1
    
    c2 = 'timedelta_rev_'+count_keys_
    df[c2] = df.click_time.diff(-1).dt.seconds.abs()
    df['key_match'] = ( df[count_keys]==df[count_keys].shift(-1) ).all(1)*1
    df.loc[df.key_match==0, c2] = -1
    
    df.drop(count_keys, axis=1, inplace=True)
    df.sort_index(inplace=True)
    
    df.iloc[0:184903890][[c1, c2]].to_pickle('../data/004__{}_train.p'.format(count_keys_))
    df.iloc[184903890:][[c1, c2]].to_pickle('../data/004__{}_test.p'.format(count_keys_))


pool = Pool(nthread)
callback = pool.map(multi, utils.comb)
pool.close()

del trte; gc.collect()

# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/004__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/004_train', 10)

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/004__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/004_test', 10)

os.system('rm -rf ../data/004__*.p')



#==============================================================================
utils.end(__file__)

