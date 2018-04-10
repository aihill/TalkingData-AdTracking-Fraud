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
nthread = 15
from collections import defaultdict
import utils
utils.start(__file__)

os.system('rm -rf ../data/003*')

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
    result = []
    for values in trte[keys].values:
        di = dict(zip(keys, values))
        key = '-'.join(map(str, [di[k] for k in count_keys]))
        
        if key in click_history:
            result.append( (di['click_time'] - click_history[key]).seconds )
        else:
            result.append(-1)
        
        click_history[key] = di['click_time']
    
    result = pd.DataFrame(result, columns=['timedelta_'+count_keys_])
    
    result.iloc[0:184903890].to_pickle('../data/003__{}_train.p'.format(count_keys_))
    result.iloc[184903890:].to_pickle('../data/003__{}_test.p'.format(count_keys_))


pool = Pool(nthread)
callback = pool.map(multi, utils.comb)
pool.close()

del trte; gc.collect()

# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/003__*_train.p'))], axis=1)
utils.to_pickles(df, '../data/003_train', 10)

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/003__*_test.p'))], axis=1)
utils.to_pickles(df, '../data/003_test', 10)

os.system('rm -rf ../data/003__*.p')



#==============================================================================
utils.end(__file__)

