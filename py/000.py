#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:09:14 2018

@author: kazuki.onodera

"""

import pandas as pd
#from glob import glob
#from multiprocessing import Pool
#total_proc = 4
import os
import gc
import utils

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
# train
# =============================================================================

print('loading train...')
train = pd.read_csv('../input/train.csv.zip', dtype=dtypes, 
                    parse_dates=['click_time', 'attributed_time']) # not date_parser
print('finish loading!')

utils.to_pickles(train, '../data/train', 10)

del train; gc.collect()

# =============================================================================
# test
# =============================================================================

print('loading test_old...')
test = pd.read_csv('../input/test_old.csv.gz', dtype=dtypes,
                   parse_dates=['click_time'])
print('finish loading!')

utils.to_pickles(test,  '../data/test_old',  10)



print('loading test...')
test = pd.read_csv('../input/test.csv.zip', dtype=dtypes,
                   parse_dates=['click_time'])
print('finish loading!')

utils.to_pickles(test,  '../data/test',  10)



