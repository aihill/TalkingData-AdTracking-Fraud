#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 18:00:44 2018

@author: kazuki.onodera
"""

import pandas as pd
from multiprocessing import Pool
import utils

print('loading train...')
train = utils.read_pickles('../data/train', col=['ip', 'app', 'device', 'os', 'channel'])

print('loading test...')
test = utils.read_pickles('../data/test_old', col=['ip', 'app', 'device', 'os', 'channel'])

print('finish loading!')


# =============================================================================
# def
# =============================================================================
def multi(c):
    
    print(c)
    
    df_tr = train[c].value_counts().to_frame()#.reset_index()
    df_tr.columns = ['cnt_train']
    
    df_te = test[c].value_counts().to_frame()#.reset_index()
    df_te.columns = ['cnt_test']
    
    df = pd.concat([df_tr, df_te], axis=1).reset_index()
    df.columns = [c, 'cnt_train', 'cnt_test']
    df = df[[c]]
    
    binary = df.index.map(bin)
    length = binary.map(len).max() -1
    binary = binary.map(lambda x: x[2:].zfill(length))
    
    for i in range(length):
        df['{}_binary_{}'.format(c, i)] = binary.map(lambda x: int(x[i]))
    
    
    df.to_pickle('../data/{}_binary.p'.format(c))

# =============================================================================
# main
# =============================================================================

li = ['ip', 'app', 'device', 'os', 'channel']
pool = Pool(len(li))
callback = pool.map(multi, li)
pool.close()









