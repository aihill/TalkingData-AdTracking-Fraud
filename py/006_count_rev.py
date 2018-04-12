#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 01:52:05 2018

@author: Kazuki
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

os.system('rm -rf ../data/006*')

trte = pd.concat([utils.read_pickles('../data/test_old_rev', ['ip', 'app', 'device', 'os', 'channel', 'click_time']),
                  utils.read_pickles('../data/train_rev', ['ip', 'app', 'device', 'os', 'channel', 'click_time'])])


def multi(count_keys):
    """
    ex:
    count_keys = ('app', 'device')
    
    """
    gc.collect()
    print(count_keys)
    
    count_keys_ = '-'.join(count_keys)
    counter = defaultdict(int)
    keys = ['ip', 'app', 'device', 'os', 'channel']
    result = []
    for values in trte[keys].values:
        di = dict(zip(keys, values))
        key = '-'.join(map(str, [di[k] for k in count_keys]))
        
        result.append(counter[key])
        counter[key] +=1
    
    result = pd.DataFrame(result, columns=['count_rev_'+count_keys_])
    
    result.iloc[0:58175885].to_pickle('../data/006__{}_test.p'.format(count_keys_))
    result.iloc[58175885:].to_pickle('../data/006__{}_train.p'.format(count_keys_))


pool = Pool(nthread)
callback = pool.map(multi, utils.comb)
pool.close()

del trte; gc.collect()

# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/006__*_train.p'))], axis=1)
utils.to_pickles(df, '../data/006_train', 10)


# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/006__*_test.p'))], axis=1)
utils.to_pickles(df, '../data/006_test', 10)

os.system('rm -rf ../data/006__*.p')


#==============================================================================
utils.end(__file__)


