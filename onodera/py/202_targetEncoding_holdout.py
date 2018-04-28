#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 09:47:33 2018

@author: Kazuki
"""

import os
import pandas as pd
from tqdm import tqdm
import gc
from glob import glob
import numpy as np
from multiprocessing import Pool
import utils
utils.start(__file__)


nthread = 10
threshold = 300

np.random.seed(71)

os.system('rm -rf ../data/202__*.p')


print("""
# =============================================================================
# mk base tbl
# =============================================================================
""")
train = utils.read_pickles('../data/train', ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'])
train['fold'] = np.random.randint(5, size=train.shape[0])


def multi_mk202(keys):
    print(keys)
    keys = list(keys)
    keys_ = '-'.join(keys)
    c = f'targetEncoding_ho_{keys_}'
        
    gr = train.groupby(keys+['fold'])
    sum_ = gr['is_attributed'].sum()
    gc.collect()
    size = gr.size()
    gc.collect()
    df = pd.concat([sum_, size], axis=1)
    df.columns = ['a', 'b']
    df[c] = df.a/df.b
    df['target_mean'] = df[c]
    
    df['totalsize'] = df.groupby(keys)['b'].transform('sum')
    
    df_upper = df[df.totalsize>=threshold]
    df_lower = df[df.totalsize<threshold]
    df_lower[c] = df_lower.groupby('fold')[c].transform('mean')
    df_lower['target_mean'] = df[c]
    
    df = pd.concat([df_upper, df_lower])[[c]].reset_index()
    
    df.to_pickle(f'../data/202__{c}.p')

pool = Pool(nthread)
callback = pool.map(multi_mk202, utils.comb)
pool.close()


print("""
# =============================================================================
# merge
# =============================================================================
""")

test = utils.read_pickles('../data/test_old', ['ip', 'app', 'device', 'os', 'channel'])
test['fold'] = np.random.randint(5, size=test.shape[0])

trte = pd.concat([train[['ip', 'app', 'device', 'os', 'channel']], test],
                 ignore_index=True)

def multi_merge(keys):
    """
    ex:
    keys = ('app', 'device')
    
    """
    gc.collect()
    print(keys)
    keys = list(keys)
    keys_ = '-'.join(keys)
    c = f'targetEncoding_ho_{keys_}'
    
    df = pd.read_pickle(f'../data/202__{c}.p')
    
    result = pd.merge(trte, df, on=keys+['fold'], how='left')
    
    result.iloc[0:utils.TRAIN_SHAPE][c].to_pickle(f'../data/202__{keys_}_train.p')
    result.iloc[utils.TRAIN_SHAPE:][c].to_pickle(f'../data/202__{keys_}_test.p')
    gc.collect()
    
    return


pool = Pool(nthread)
callback = pool.map(multi_merge, utils.comb)
pool.close()

del trte; gc.collect()

print("""
# =============================================================================
# concat
# =============================================================================
""")

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/202__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/202_train', utils.SPLIT_SIZE)

del df; gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/202__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/202_test', utils.SPLIT_SIZE)

os.system('rm -rf ../data/202__*.p')


#==============================================================================
utils.end(__file__)





