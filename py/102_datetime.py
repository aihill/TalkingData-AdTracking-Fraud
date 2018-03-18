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

max_time = test.click_time.max()

# =============================================================================
# for train
# =============================================================================


train['hour'] = train.click_time.dt.hour + (train.click_time.dt.minute/60)
train['timestamp'] = (max_time - train.click_time).seconds


col = ['hour', 'timestamp']
train[col].to_pickle('../data/102_train.p')

del train; gc.collect()


# =============================================================================
# for test
# =============================================================================

test['hour'] = test.click_time.dt.hour + (test.click_time.dt.minute/60)
test['timestamp'] = (max_time - test.click_time).seconds


col = ['hour', 'timestamp']
test[col].to_pickle('../data/102_test_old.p')


