#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 18:43:43 2018

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
from collections import defaultdict
import utils
utils.start(__file__)

os.system('rm -rf ../data/005*')

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
    click_history = defaultdict(int)
    keys = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
    click_time_bk = key_bk = None
    cnt = 0
    result = []
    for values in (trte[keys].values):
        di = dict(zip(keys, values))
        key = '-'.join(map(str, [di[k] for k in count_keys]))
        
        if click_time_bk is None:
            pass
        elif click_time_bk == di['click_time'] and key_bk == key:
            cnt +=1
            if cnt > click_history[key]:
                click_history[key] = cnt
        else:
            cnt = 0
            
        result.append(click_history[key])
        click_time_bk = di['click_time']
        key_bk = key
        
        
    result = pd.DataFrame(result, columns=['sametime_maxcount_'+count_keys_])
    
    
    result.iloc[0:184903890].to_pickle('../data/005__{}_train.p'.format(count_keys_))
    result.iloc[184903890:].to_pickle('../data/005__{}_test.p'.format(count_keys_))


pool = Pool(nthread)
callback = pool.map(multi, utils.comb)
pool.close()

del trte; gc.collect()

# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/005__*_train.p'))], axis=1)
utils.to_pickles(df, '../data/005_train', 10)


# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/005__*_test.p'))], axis=1)
utils.to_pickles(df, '../data/005_test', 10)

os.system('rm -rf ../data/005__*.p')




#==============================================================================
utils.end(__file__)


