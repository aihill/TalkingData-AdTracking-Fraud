#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 18:00:44 2018

@author: kazuki.onodera
"""

import pandas as pd
import utils


train = utils.read_pickles('../data/train').sort_values('click_time')
test = utils.read_pickles('../data/test').sort_values('click_time')

# =============================================================================
# channel
# =============================================================================

channel_tr = train.channel.value_counts().to_frame()#.reset_index()
channel_tr.columns = ['cnt_train']

channel_te = test.channel.value_counts().to_frame()#.reset_index()
channel_te.columns = ['cnt_test']

channel = pd.concat([channel_tr, channel_te], axis=1)


channel['binary'] = channel.index.map(bin)
channel.binary = channel.binary.map(lambda x: x[2:].zfill(9))

for i in range(9):
    channel['binary_{}'.format(i)] = channel.binary.map(lambda x: int(x[i]))


channel.fillna(0).to_pickle('../data/channel.p')


