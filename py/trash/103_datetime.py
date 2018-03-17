#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 03:40:04 2018

@author: Kazuki

datetime feature

"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import utils

# =============================================================================
# for valid
# =============================================================================
valid = utils.read_pickles('../data/valid').sort_values(utils.sort_keys) # be sure to sort by this keys

valid['hour'] = valid.click_time.dt.hour + (valid.click_time.dt.minute/60)

valid[['hour']].to_pickle('../data/103_valid.p')

del valid; gc.collect()

# =============================================================================
# for test
# =============================================================================
test = utils.read_pickles('../data/test').sort_values(utils.sort_keys)

test['hour'] = test.click_time.dt.hour + (test.click_time.dt.minute/60)

test[['hour']].to_pickle('../data/103_test.p')
