#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:09:14 2018

@author: kazuki.onodera
"""

import pandas as pd
from glob import glob
from multiprocessing import Pool
total_proc = 4
import utils

dtypes = {
        'ip':'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
        }


print('load')
train = pd.read_csv('../input/train.csv.zip', dtype=dtypes, 
                    parse_dates=['click_time', 'attributed_time']) # date_parser is not effective, don't know why
test = pd.read_csv('../input/test.csv.zip', dtype=dtypes,
                   parse_dates=['click_time'])

# =============================================================================
# multiprocessing
# =============================================================================

#def convert_date(args):
#    df = args[0]
#    c = args[1]
#    df[c] = df[c].map(pd.to_datetime)
#
#
#pool = Pool(total_proc)
#
#pool.map(convert_date, [(train, 'click_time'), (train, 'attributed_time'), (test, 'click_time'), (test, 'attributed_time')])
#
#pool.close()

# =============================================================================
# not multiprocessing
# =============================================================================

#train.click_time = train.click_time.map(pd.to_datetime)
#train.attributed_time = train.attributed_time.map(pd.to_datetime)
#
#test.click_time = test.click_time.map(pd.to_datetime)
#test.attributed_time = test.attributed_time.map(pd.to_datetime)
#
#


utils.to_pickles(train, '../data/train', 10)
utils.to_pickles(test,  '../data/test',  10)



