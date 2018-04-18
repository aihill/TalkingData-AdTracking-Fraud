#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:40:52 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import utils

train = utils.read_pickles('../data/train')
test  = utils.read_pickles('../data/test_old')

min_time = train.click_time.min()


# =============================================================================
# def
# =============================================================================
def multi(p):
    if p==0:
        train['hour'] = train.click_time.dt.hour + (train.click_time.dt.minute/60)
        train['timestamp'] = (train.click_time - min_time).dt.seconds
        
        
        col = ['hour', 'timestamp']
        train[col].to_pickle('../data/102_train.p')
        
        del train; gc.collect()
        
    elif p==1:
        test['hour'] = test.click_time.dt.hour + (test.click_time.dt.minute/60)
        test['timestamp'] = (test.click_time - min_time).dt.seconds
        
        col = ['hour', 'timestamp']
        test[col].to_pickle('../data/102_test_old.p')


# =============================================================================
# main
# =============================================================================



pool = Pool(2)
callback = pool.map(multi, range(2))
pool.close()
