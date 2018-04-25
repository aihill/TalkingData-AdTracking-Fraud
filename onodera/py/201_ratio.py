#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 01:38:57 2018

@author: Kazuki
"""


import os
import pandas as pd
from tqdm import tqdm
import gc
from glob import glob
from multiprocessing import Pool
import utils
utils.start(__file__)


nthread = 5
threshold = 300

os.system('rm -rf ../data/201__*.p')

train = utils.read_pickles('../data/train')

trte = pd.concat([train, 
                  utils.read_pickles('../data/test_old')])


def multi(keys):
    """
    ex:
    keys = ('app', 'device')
    
    """
    gc.collect()
    print(keys)
    keys = list(keys)
    keys_ = '-'.join(keys)
    c = f'targetEncoding_{keys_}'
    
    gr = train.groupby(keys)
    sum_ = gr['is_attributed'].sum()
    size = gr.size()
    df = pd.concat([sum_, size], axis=1)
    df.columns = ['a', 'b']
    df[c] = df.a/df.b
    
    df_upper = df[df['b']>=threshold]
    df_lower = df[df['b']<threshold]
    df_lower[c] = df_lower.a.sum() / df_lower.b.sum()
    
    df = pd.concat([df_upper, df_lower])[c].reset_index()
    
    result = pd.merge(trte, df, on=keys, how='left')
    
    result.iloc[0:utils.TRAIN_SHAPE][c].to_pickle(f'../data/201__{keys_}_train.p')
    result.iloc[utils.TRAIN_SHAPE:][c].to_pickle(f'../data/201__{keys_}_test.p')
    gc.collect()
    
    return

# =============================================================================
# 
# =============================================================================
pool = Pool(nthread)
callback = pool.map(multi, utils.comb)
pool.close()

del trte; gc.collect()

# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/201__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/201_train', utils.SPLIT_SIZE)


# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/201__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/201_test', utils.SPLIT_SIZE)

os.system('rm -rf ../data/201__*.p')

