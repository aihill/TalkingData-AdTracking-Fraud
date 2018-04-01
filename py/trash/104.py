#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 06:44:53 2018

@author: Kazuki
"""

from glob import glob
import pandas as pd
from tqdm import tqdm
import gc
from os import system
from multiprocessing import Pool
import utils
utils.start(__file__)

proc = 6

# =============================================================================
# train
# =============================================================================
train = utils.read_pickles('../data/train')

def multi_train(keys):
    """
    keys = ['ip', 'device', 'os']
    """
    gc.collect()
    keys_ = '-'.join(keys)
    print(keys)
    df_ = train.sort_values(list(keys) + ['click_time'])
    
    key_values_bk = click_time_bk = None
    time_deltas = []
    cnts = []
    cnt = 0
    for values in df_[list(keys) + ['click_time']].values:
        
        key_values = list(values[:-1])
        click_time = values[-1]
        
        if key_values_bk is None:
            time_deltas.append(-1)
            cnt = 0
            cnts.append(cnt)
            
        elif key_values==key_values_bk:
            time_deltas.append((click_time - click_time_bk).seconds)
            cnt +=1
            cnts.append(cnt)
            
        else:
            time_deltas.append(-1)
            cnt = 0
            cnts.append(cnt)
        
        key_values_bk, click_time_bk = key_values, click_time
    
    c1 = '{}_time_delta'.format(keys_)
    c2 = '{}_sequence_count'.format(keys_)
    df_[c1] = time_deltas
    df_[c2] = cnts
    
    df_.sort_values(utils.sort_keys)[[c1, c2]].to_pickle('../data/104_train_{}.p'.format(keys_))



pool = Pool(proc)
callback = pool.map(multi_train, utils.comb)
pool.close()

del train; gc.collect()


# =============================================================================
# test
# =============================================================================

test  = utils.read_pickles('../data/test_old')


def multi_test(keys):
    
    gc.collect()
    keys_ = '-'.join(keys)
    print(keys)
    df_ = test.sort_values(list(keys) + ['click_time'])
    
    key_values_bk = click_time_bk = None
    time_deltas = []
    cnts = []
    cnt = 0
    for values in df_[list(keys) + ['click_time']].values:
        
        key_values = list(values[:-1])
        click_time = values[-1]
        
        if key_values_bk is None:
            time_deltas.append(-1)
            cnt = 0
            cnts.append(cnt)
            
        elif key_values==key_values_bk:
            time_deltas.append((click_time - click_time_bk).seconds)
            cnt +=1
            cnts.append(cnt)
            
        else:
            time_deltas.append(-1)
            cnt = 0
            cnts.append(cnt)
        
        key_values_bk, click_time_bk = key_values, click_time
    
    c1 = '{}_time_delta'.format(keys_)
    c2 = '{}_sequence_count'.format(keys_)
    df_[c1] = time_deltas
    df_[c2] = cnts
    
    df_.sort_values(utils.sort_keys)[[c1, c2]].to_pickle('../data/104_test_{}.p'.format(keys_))


pool = Pool(proc)
callback = pool.map(multi_test, utils.comb)
pool.close()

del test; gc.collect()

# =============================================================================
# concat
# =============================================================================


print('concat train')
pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob('../data/104_train_*.p')))], axis=1).to_pickle('../data/104_train.p')
system('rm ../data/104_train_*')

gc.collect()

print('concat test')
pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob('../data/104_test_*.p')))], axis=1).to_pickle('../data/104_test.p')
system('rm ../data/104_test_*')


#==============================================================================
utils.end(__file__)

