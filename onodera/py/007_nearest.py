#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 22:26:23 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
from time import time
import gc
import os
from glob import glob
from multiprocessing import Pool
nthread = 5
from itertools import combinations
import utils
utils.start(__file__)



trte = pd.concat([utils.read_pickles('../data/train', ['ip', 'app', 'device', 'os', 'channel', 'click_time']),
                utils.read_pickles('../data/test_old', ['ip', 'app', 'device', 'os', 'channel', 'click_time'])],
                ignore_index=True)

gc.collect()

col = []
for c in ['ip', 'app', 'device', 'os', 'channel']:
    trte[f'nearestNext_{c}'] = trte[c].shift(-1)
    trte[f'nearestPre_{c}'] = trte[c].shift()
    col.append(f'nearestNext_{c}')
    col.append(f'nearestPre_{c}')
    gc.collect()

train = trte.iloc[0:utils.TRAIN_SHAPE][col].reset_index(drop=True)
test  = trte.iloc[utils.TRAIN_SHAPE:][col].reset_index(drop=True)

del trte; gc.collect()

# =============================================================================
# concat
# =============================================================================

# train
utils.to_pickles(train, '../data/007_train', utils.SPLIT_SIZE)

print(train.columns.tolist())
del train; gc.collect()

# test
utils.to_pickles(test, '../data/007_test', utils.SPLIT_SIZE)




#==============================================================================
utils.end(__file__)


