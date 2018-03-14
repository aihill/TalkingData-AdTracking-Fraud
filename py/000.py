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
# validation( train -> valid_feature, valid)
# =============================================================================

print('loading train...')
train = pd.read_csv('../input/train.csv.zip', dtype=dtypes, 
                    parse_dates=['click_time', 'attributed_time']) # not date_parser
print('finish loading!')

valid_feature = train[train.click_time<=pd.to_datetime('2017-11-08 17:00:00')] # TODO: consider datetime
valid = train[train.click_time>=pd.to_datetime('2017-11-09 05:00:00')]

utils.to_pickles(valid_feature, '../data/valid_feature', 10)
utils.to_pickles(valid, '../data/valid', 10)

del valid_feature, valid; gc.collect()

# =============================================================================
# test
# =============================================================================

print('loading test...')
test = pd.read_csv('../input/test.csv.zip', dtype=dtypes,
                   parse_dates=['click_time'])
print('finish loading!')

test_feature = train[train.click_time>=pd.to_datetime('2017-11-07 13:30:00')]

utils.to_pickles(test_feature, '../data/test_feature', 10)
utils.to_pickles(test,  '../data/test',  10)



