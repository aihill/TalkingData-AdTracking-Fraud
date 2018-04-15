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
nthread = 5
#from collections import defaultdict
import utils
utils.start(__file__)

os.system('rm -rf ../data/004*')

trte = pd.concat([utils.read_pickles('../data/train', ['ip', 'app', 'device', 'os', 'channel', 'click_time']),
                utils.read_pickles('../data/test_old', ['ip', 'app', 'device', 'os', 'channel', 'click_time'])])

def multi(count_keys):
    """
    ex:
    count_keys = ('app', 'device')
    
    """
    
    gc.collect()
    print(count_keys)
    
    count_keys_ = '-'.join(count_keys)
    keys = count_keys+['click_time']
    df = trte[keys].sort_values(keys)
    result = []
    click_bk = key_bk = None
    for values in (df.values):
        di = dict(zip(keys, values))
        key = '-'.join(map(str, [di[k] for k in count_keys]))
        
        if key_bk is None:
            result.append(-1)
        elif key == key_bk:
            result.append( (di['click_time'] - click_bk).seconds )
        else:
            result.append(-1)
        
        key_bk = key
        click_bk = di['click_time']
    
    c = 'timedelta_'+count_keys_
    df[c] = result
    df.sort_values('click_time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df.iloc[0:184903890][c].to_pickle('../data/004__{}_train.p'.format(count_keys_))
    df.iloc[184903890:][c].to_pickle('../data/004__{}_test.p'.format(count_keys_))


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

