#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 18:17:36 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from multiprocessing import Pool
import utils
utils.start(__file__)

train = utils.read_pickles('../data/train', ['click_time'])
test  = utils.read_pickles('../data/test_old', ['click_time'])

min_time = train.click_time.min()


# =============================================================================
# def
# =============================================================================
def multi(p):
    if p==0:
        train['day'] = train.click_time.dt.day
        train['hour'] = train.click_time.dt.hour
        train['hour_min'] = train['hour'] + (train.click_time.dt.minute/60)
        train['timestamp'] = (train.click_time - min_time).dt.seconds
        
        col = ['hour', 'timestamp']
        utils.to_pickles(train[col], '../data/001_train', 10)
                
    elif p==1:
        test['day']  = test.click_time.dt.day
        test['hour'] = test.click_time.dt.hour
        test['hour_min'] = test['hour'] + (test.click_time.dt.minute/60)
        test['timestamp'] = (test.click_time - min_time).dt.seconds
        
        col = ['hour', 'timestamp']
        utils.to_pickles(test[col], '../data/001_test', 10)


# =============================================================================
# main
# =============================================================================



pool = Pool(2)
callback = pool.map(multi, range(2))
pool.close()



#==============================================================================
utils.end(__file__)

