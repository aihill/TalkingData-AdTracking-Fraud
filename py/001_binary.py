#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 18:00:44 2018

@author: kazuki.onodera
"""

import pandas as pd
import utils

dtypes = {
        'ip':'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
        }


print('loading train...')
train = pd.read_csv('../input/train.csv.zip', dtype=dtypes, 
                    usecols=['ip', 'app', 'device', 'os', 'channel']) # not date_parser
print('loading test...')
test = pd.read_csv('../input/test.csv.zip', dtype=dtypes,
                   usecols=['ip', 'app', 'device', 'os', 'channel'])
print('finish loading!')


# =============================================================================
# def
# =============================================================================
def main(c):
    
    print(c)
    
    df_tr = train[c].value_counts().to_frame()#.reset_index()
    df_tr.columns = ['cnt_train']
    
    df_te = test[c].value_counts().to_frame()#.reset_index()
    df_te.columns = ['cnt_test']
    
    df = pd.concat([df_tr, df_te], axis=1).reset_index()
    df.columns = [c, 'cnt_train', 'cnt_test']
    df = df[[c]]
    
    df['binary'] = df.index.map(bin)
    length = df['binary'].map(len).max() -1
    df.binary = df.binary.map(lambda x: x[2:].zfill(length))
    
    for i in range(length):
        df['{}_binary_{}'.format(c, i)] = df.binary.map(lambda x: int(x[i]))
    
    
    df.to_pickle('../data/{}.p'.format(c))

# =============================================================================
# 
# =============================================================================

main('ip')
main('app')
main('device')
main('os')
main('channel')











