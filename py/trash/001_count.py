#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 22:10:35 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from multiprocessing import Pool
from collections import defaultdict
import utils
utils.start(__file__)


trte = pd.concat([utils.read_pickles('../data/train', ['ip', 'app', 'device', 'os', 'channel', 'click_time']),
                utils.read_pickles('../data/test_old', ['ip', 'app', 'device', 'os', 'channel', 'click_time'])])

ip_counter = defaultdict(int)
app_counter = defaultdict(int)
device_counter = defaultdict(int)
os_counter = defaultdict(int)
channel_counter = defaultdict(int)

ip_result = []
app_result = []
device_result = []
os_result = []
channel_result = []

for values in tqdm(trte[['ip', 'app', 'device', 'os', 'channel']].values):
    ip, app, device, os, channel = values
    
    ip_result.append(ip_counter[ip])
    app_result.append(app_counter[app])
    device_result.append(device_counter[device])
    os_result.append(os_counter[os])
    channel_result.append(channel_counter[channel])
    
    ip_counter[ip] +=1
    app_counter[app] +=1
    device_counter[device] +=1
    os_counter[os] +=1
    channel_counter[channel] +=1
    
184903890

trte['count_ip'] = ip_result
trte['count_app'] = app_result
trte['count_device'] = device_result
trte['count_os'] = os_result
trte['count_channel'] = channel_result


trte.iloc[0:184903890][['count_ip', 'count_app', 'count_device', 'count_os', 'count_channel']].to_pickle('../data/001_train.p')
trte.iloc[184903890:][['count_ip', 'count_app', 'count_device', 'count_os', 'count_channel']].to_pickle('../data/001_test.p')



