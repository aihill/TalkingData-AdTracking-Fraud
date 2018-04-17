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

os.system('rm -rf ../data/109__*.p')

trte = pd.concat([utils.read_pickles('../data/train'), 
                  utils.read_pickles('../data/test_old')])

trte['day']  = trte.click_time.dt.day
trte['hour'] = trte.click_time.dt.hour

gc.collect()



def multi(keys):
    gc.collect()
    print(keys)
    keys1, keys2 = keys
    
    df = trte.groupby(keys1).size().groupby(keys2).size().rank(method='dense')
    c = 'nunique_' + '-'.join(keys1) + '_' + '-'.join(keys2)
    df.name = c
    df = df.reset_index()
    utils.reduce_memory(df, ix_start=-1)
    
    result = pd.merge(trte, df, on=keys2, how='left')
    
    result.iloc[0:utils.TRAIN_SHAPE][c].to_pickle('../data/109__{}_train.p'.format(c))
    result.iloc[utils.TRAIN_SHAPE:][c].to_pickle('../data/109__{}_test.p'.format(c))
    gc.collect()


comb = []
for i in range(2, 8):
    tmp1 = list(combinations(['ip', 'app', 'device', 'os', 'channel', 'day', 'hour'], i))
    for c1 in tmp1:
        if 'hour' not in c1 and 'day' not in c1:
            continue
        for j in range(1, min(4, i)):
            tmp2 = list(combinations(c1, j))
            for c2 in tmp2:
                if ('hour',)==c2 or ('day',)==c2 or ('device', 'os')==c2 or ('os',)==c2 or ('device',)==c2 :
                    continue
                comb.append( [list(c1), list(c2)] )


# =============================================================================
# concat pt1
# =============================================================================
st = 0
end = 0
limit = 10
for pt in range(1,4):
    end +=limit
    print(st, end)
    gc.collect()
    pool = Pool(proc)
    callback = pool.map(multi, comb[st:end])
    pool.close()
    st = end
    
    # train
    df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/109__*_train.p'))], axis=1).reset_index(drop=True)
    utils.to_pickles(df, '../data/109-{}_train'.format(pt), 10)
    
    gc.collect()
    
    # test
    df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/109__*_test.p'))], axis=1).reset_index(drop=True)
    utils.to_pickles(df, '../data/109-{}_test'.format(pt), 10)
    
    os.system('rm -rf ../data/109__*.p')


#==============================================================================
utils.end(__file__)


