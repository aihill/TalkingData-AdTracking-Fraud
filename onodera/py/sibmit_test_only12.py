#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 15:24:17 2018

@author: kazuki.onodera
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lightgbm as lgb
import os
import gc
from time import sleep
import utils

IN_SUBMIT_FILE_PATH = '../output/repro429-2.csv.gz'
OUT_SUBMIT_FILE_PATH = '../output/repro429-2_only12.csv.gz'
COMMENT = 'only click_time 12'


sub = pd.read_csv(IN_SUBMIT_FILE_PATH)

click_id = pd.read_pickle('../data/sub_429-2.p')

test = utils.read_pickles('../data/test_old')

test = test.drop_duplicates('click_id', keep='first')
test = test[~test.click_id.isnull()]
test.click_id = test.click_id.map(int)


df = pd.merge(sub, test[['click_id', 'click_time']], on='click_id', how='left')
df.click_time += pd.offsets.Hour(8)

df['hour_min'] = df.click_time.dt.hour + (df.click_time.dt.minute/60).round(1)

df['hour_min'].value_counts()

df['hour'] = df.click_time.dt.hour

df.loc[df.hour!=12, 'is_attributed'] = 0

sub = df[['click_id', 'is_attributed']]

# =============================================================================
# submission
# =============================================================================
sub.to_csv(OUT_SUBMIT_FILE_PATH, index=False, compression='gzip')
print('submit')
utils.submit(OUT_SUBMIT_FILE_PATH, COMMENT)


