#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 22:09:59 2018

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

os.system('rm -rf ../data/114__*.p')

trte = pd.concat([utils.read_pickles('../data/train'),
                utils.read_pickles('../data/test_old')])
trte['hour']  = (trte.click_time + pd.offsets.Hour(8)).dt.hour

def multi(k):
    """
    k = 'app'
    """
    gc.collect()
    print(k)
    
    df = pd.crosstab(trte[k], trte.hour, normalize='index')
    df = df.add_prefix(f'histHour_{k}_')
    
    utils.reduce_memory(df)
    col = df.columns.tolist()
    
    result = pd.merge(trte, df, on=k, how='left')
    gc.collect()
    
#    result.iloc[0:utils.TRAIN_SHAPE][col].to_pickle(f'../data/114__{k}_train.p')
#    result.iloc[utils.TRAIN_SHAPE:][col].to_pickle(f'../data/114__{k}_test.p')
#    gc.collect()
    
    utils.to_pickles(result.iloc[0:utils.TRAIN_SHAPE][col].reset_index(drop=True), 
                     '../data/114_train', utils.SPLIT_SIZE)
    gc.collect()
    utils.to_pickles(result.iloc[utils.TRAIN_SHAPE:][col].reset_index(drop=True), 
                     '../data/114_test', utils.SPLIT_SIZE)
    
    

#pool = Pool(5)
#callback = pool.map(multi, ['ip', 'app', 'device', 'os', 'channel'])
#pool.close()

multi('ip')

# =============================================================================
# concat
# =============================================================================

## train
#df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/114__*_train.p'))], axis=1).reset_index(drop=True)
#utils.to_pickles(df, '../data/114_train', utils.SPLIT_SIZE)
#
#gc.collect()
#
## test
#df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/114__*_test.p'))], axis=1).reset_index(drop=True)
#utils.to_pickles(df, '../data/114_test', utils.SPLIT_SIZE)
#
#os.system('rm -rf ../data/114__*.p')



#==============================================================================
utils.end(__file__)


