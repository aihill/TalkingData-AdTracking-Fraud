#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:42:02 2018

@author: kazuki.onodera


concatして使う

"""

from glob import glob
import pandas as pd
from tqdm import tqdm
import gc
from multiprocessing import Pool
import utils
#utils.start(__file__)


train = utils.read_pickles('../data/train')
test  = utils.read_pickles('../data/test_old')


def multi(keys):
    keys_ = '-'.join(keys)
    df1 = train.groupby('ip').is_attributed.sum()
    df1.name = 'label_enc_sum'
    
    df2 = train.groupby('ip').size()
    df2.name = 'label_enc_count'
    
    df = pd.concat([df1, df2], axis=1)
    df.reset_index(inplace=True)
    
    train_ = pd.merge(train, df, on=keys, how='left')
    train_['label_enc_count'] -=1
    train_['label_enc_sum'] -= train_['is_attributed']
    train_['label_enc_ratio'] = train_['label_enc_sum'] / train_['label_enc_count']
    train_[['label_enc_sum', 'label_enc_ratio']].add_prefix(keys_).to_pickle('../data/103_train_{}_label_enc.p'.format(keys_))
    gc.collect()
    
    test_ = pd.merge(test, df, on=keys, how='left')
    test_['label_enc_ratio'] = test_['label_enc_sum'] / test_['label_enc_count']
    test_[['label_enc_sum', 'label_enc_ratio']].add_prefix(keys_).to_pickle('../data/103_test_{}_label_enc.p'.format(keys_))
    gc.collect()


pool = Pool(16)
callback = pool.map(multi, utils.comb)
pool.close()


print('concat train')
pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob('../data/103_train_*_label_enc.p')))], axis=1).to_pickle('../data/103_train.p')

print('concat test')
pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob('../data/103_test_*_label_enc.p')))], axis=1).to_pickle('../data/103_test.p')



