#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:09:14 2018

@author: kazuki.onodera

"""

import pandas as pd
#from glob import glob
from multiprocessing import Pool
#total_proc = 4
import os
import gc
import utils
utils.start(__file__)

# setting


os.system('rm -rf ../data')
os.system('mkdir ../data')

dtypes = {
        'ip':'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
        }

# =============================================================================
# def
# =============================================================================
def multi(p):
    if p==0:
        # =============================================================================
        # test
        # =============================================================================
        
        print('loading test_old...')
        test_old = pd.read_csv('../input/test_old.csv.gz', dtype=dtypes,
                           parse_dates=['click_time']).sort_values(utils.sort_keys) # be sure to sort by this keys
        
        print('loading test...')
        test = pd.read_csv('../input/test.csv.zip', dtype=dtypes,
                           parse_dates=['click_time']).sort_values(utils.sort_keys).reset_index(drop=True)
        print('finish loading!')
        
        merge_key = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
        test_old.drop('click_id', axis=1, inplace=True)
        test_old = pd.merge(test_old, test[merge_key+['click_id']], on=merge_key, how='left')
        
        utils.to_pickles(test_old,  '../data/test_old',  utils.SPLIT_SIZE)
#        utils.to_pickles(test_old.sort_values(utils.sort_keys, ascending=False),
#                         '../data/test_old_rev',  10)
        utils.to_pickles(test,  '../data/test',  utils.SPLIT_SIZE)
        
        del test_old, test; gc.collect()
    
    else:
        # =============================================================================
        # train
        # =============================================================================
        
        print('loading train...')
        train = pd.read_csv('../input/train.csv.zip', dtype=dtypes, 
                            parse_dates=['click_time', 'attributed_time']).sort_values(utils.sort_keys) # be sure to sort by this keys
        print('finish loading!')
        
#        print('drop os; 607, 748, 866')
#        train = train[~train.os.isin([607, 748, 866])].reset_index(drop=True)
        print('train.shape', train.shape)
        utils.to_pickles(train, '../data/train', utils.SPLIT_SIZE)
        utils.to_pickles(train.is_attributed, '../data/is_attributed', utils.SPLIT_SIZE)
#        utils.to_pickles(train.sort_values(utils.sort_keys, ascending=False), 
#                         '../data/train_rev', 10)
        
        del train; gc.collect()
# =============================================================================
# main
# =============================================================================

pool = Pool(2)
pool.map(multi, [0,1])
pool.close()

#==============================================================================
utils.end(__file__)


