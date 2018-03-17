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


# =============================================================================
# for train
# =============================================================================
df = utils.read_pickles('../data/train')

ip_bk = app_bk = device_bk = os_bk = channel_bk = click_time_bk = None
li = []
for ip, app, device, os, channel, click_time in tqdm(df[utils.sort_keys].values, miniters=99999):
    
    if ip_bk is None:
        li.append(-1)
    elif ip==ip_bk and app==app_bk and device==device_bk and os==os_bk and channel==channel_bk:
        li.append((click_time - click_time_bk).seconds)
    else:
        li.append(-1)
    
    ip_bk, app_bk, device_bk, os_bk, channel_bk, click_time_bk = ip, app, device, os, channel, click_time

df['same_ip-app-device-os-channel_diff'] = li
df['hour'] = df.click_time.dt.hour + (df.click_time.dt.minute/60)
df[['same_ip-app-device-os-channel_diff', 'hour']].to_pickle('../data/101_train.p')

del df; gc.collect()


# =============================================================================
# for test
# =============================================================================
df = utils.read_pickles('../data/test')

ip_bk = app_bk = device_bk = os_bk = channel_bk = click_time_bk = None
li = []
for ip, app, device, os, channel, click_time in tqdm(df[utils.sort_keys].values, miniters=99999):
    
    if ip_bk is None:
        li.append(-1)
    elif ip==ip_bk and app==app_bk and device==device_bk and os==os_bk and channel==channel_bk:
        li.append((click_time - click_time_bk).seconds)
    else:
        li.append(-1)
    
    ip_bk, app_bk, device_bk, os_bk, channel_bk, click_time_bk = ip, app, device, os, channel, click_time

df['same_ip-app-device-os-channel_diff'] = li
df['hour'] = df.click_time.dt.hour + (df.click_time.dt.minute/60)
df[['same_ip-app-device-os-channel_diff', 'hour']].to_pickle('../data/101_test.p')


