#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 02:48:08 2018

@author: Kazuki

simple stats

"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import utils

# =============================================================================
# for valid
# =============================================================================
valid = utils.read_pickles('../data/valid_feature')

for c in ['ip', 'app', 'device', 'os', 'channel']:
    freq = pd.crosstab(valid[c], valid.is_attributed)
    freq.columns = [c+'_freq0', c+'_freq1']
    
    per  = pd.crosstab(valid[c], valid.is_attributed, normalize='index')
    per.columns = [c+'_per0', c+'_per1']
    df = pd.concat([freq, per[[c+'_per1']]], axis=1).reset_index()
    
    
    df.to_pickle('../data/102_{}_valid.p'.format(c))

del valid


# =============================================================================
# for test
# =============================================================================
test = utils.read_pickles('../data/test_feature')

for c in ['ip', 'app', 'device', 'os', 'channel']:
    freq = pd.crosstab(test[c], test.is_attributed)
    freq.columns = [c+'_freq0', c+'_freq1']
    
    per  = pd.crosstab(test[c], test.is_attributed, normalize='index')
    per.columns = [c+'_per0', c+'_per1']
    df = pd.concat([freq, per[[c+'_per1']]], axis=1).reset_index()
    
    
    df.to_pickle('../data/102_{}_test.p'.format(c))








