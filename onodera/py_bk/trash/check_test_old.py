#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 10:46:27 2018

@author: Kazuki
"""

import pandas as pd


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

print('loading test...')
test = pd.read_csv('../input/test.csv.zip', dtype=dtypes, 
                    parse_dates=['click_time']).sort_values('click_time')

print('loading old...')
test_old = pd.read_csv('../input/test_old.csv.gz', dtype=dtypes, 
                    parse_dates=['click_time']).sort_values('click_time')



def check():
    click_id, ip, app, device, os, channel, click_time = test.sample(1).values[0]
    tmp = test_old[test_old.ip==ip]
    if tmp[tmp.app==app][tmp.device==device][tmp.os==os][tmp.channel==channel][tmp.click_time==click_time].shape[0]>0:
        print('Y')
    else:
        print('N')
    
    