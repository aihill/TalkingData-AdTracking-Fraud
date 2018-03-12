#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:40:52 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from itertools import permutations
from tqdm import tqdm
import gc
import utils

col = list(permutations(['ip', 'app', 'device', 'os', 'channel']))
col3 = list(permutations(['ip', 'app', 'device', 'os', 'channel'], 3))
col2 = list(permutations(['ip', 'app', 'device', 'os', 'channel'], 2))
col1 = list(permutations(['ip', 'app', 'device', 'os', 'channel'], 1))


col += col3 + col2 + col1

keys = list(col[0])+['click_time']


# =============================================================================
# for train
# =============================================================================
train = utils.read_pickles('../data/train').sort_values(keys)

ip_bk = app_bk = device_bk = os_bk = channel_bk = click_time_bk = None
li = []
for ip, app, device, os, channel, click_time in tqdm(train[keys].values, miniters=99999):
    
    if ip_bk is None:
        li.append(-1)
    elif ip==ip_bk and app==app_bk and device==device_bk and os==os_bk and channel==channel_bk:
        li.append(click_time - click_time_bk)
    else:
        li.append(-1)
    
    ip_bk, app_bk, device_bk, os_bk, channel_bk, click_time_bk = ip, app, device, os, channel, click_time

train['same_ip-app-device-os-channel_diff'] = li

train[['same_ip-app-device-os-channel_diff']].to_pickle('../data/101_train.p')

del train; gc.collect()


# =============================================================================
# for test
# =============================================================================
test = utils.read_pickles('../data/test').sort_values(keys)

ip_bk = app_bk = device_bk = os_bk = channel_bk = click_time_bk = None
li = []
for ip, app, device, os, channel, click_time in tqdm(test[keys].values, miniters=99999):
    
    if ip_bk is None:
        li.append(-1)
    elif ip==ip_bk and app==app_bk and device==device_bk and os==os_bk and channel==channel_bk:
        li.append(click_time - click_time_bk)
    else:
        li.append(-1)
    
    ip_bk, app_bk, device_bk, os_bk, channel_bk, click_time_bk = ip, app, device, os, channel, click_time

test['same_ip-app-device-os-channel_diff'] = li

test[['same_ip-app-device-os-channel_diff']].to_pickle('../data/101_test.p')

del test; gc.collect()

