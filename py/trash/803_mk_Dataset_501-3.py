#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 21:56:04 2018

@author: Kazuki
"""

from glob import glob
import pandas as pd
from os import system
import os
from time import sleep
from tqdm import tqdm
import gc
import lightgbm as lgb
from multiprocessing import Pool
import utils
utils.start(__file__)

# setting
useimp = 80
filepath = f'imp_802_importance_501-3.py_drop_ip-device-os-channel.csv'


system('rm ../data/803_tmp*.p')
#system('rm ../data/*.mt')
system('rm SUCCESS_803')

categorical_feature = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']

# =============================================================================
# wait
# =============================================================================
while True:
    if os.path.isfile('SUCCESS_802'):
        break
    else:
        sleep(60*1)

utils.send_line('START {}'.format(__file__))

# =============================================================================
# imp
# =============================================================================
if filepath is None:
    filepath = sorted(glob('imp*.csv'))[-1]

print(f'use imp: {filepath}')

imp = pd.read_csv(filepath)

imp.set_index('index', inplace=True)


usecols = imp.head(useimp).index.tolist() + ['is_attributed']
usecols_set = set(usecols)

print(sorted(usecols))

# =============================================================================
# def
# =============================================================================

def multi_train(args):
    load_folder, i = args
    gc.collect()
#    if load_folder == '../data/dtrain/':
#        return
    print(f'loading {load_folder} ...')
    df = pd.read_pickle(load_folder + '/000.p')
    col = list(set(df.columns) & usecols_set)
    if len(col)>0:
        df = pd.concat([pd.read_pickle(load_folder + f'/{j:03d}.p')[col] for j in range(0, 100)])
        gc.collect()
        df[col].reset_index(drop=True).fillna(-1).to_pickle(f'../data/803_tmp{i}.p')

def multi_test(args):
    load_folder, i = args
    gc.collect()
    print(f'loading {load_folder} ...')
    if load_folder=='../data/test_old/':
        df = utils.read_pickles(load_folder)
        
    else:
        df = pd.read_pickle(load_folder + '/000.p')
        col = list(set(df.columns) & usecols_set)
        if len(col)==0:
            return
        df = pd.concat([sub, utils.read_pickles(load_folder, col)],
                        axis=1)
    col = list(set(df.columns) & usecols_set)
    if len(col)>0:
        df = df[~df.click_id.isnull()]
        df.drop_duplicates('click_id', keep='last', inplace=True) # last?
        print(load_folder, df.shape)
        df[col].reset_index(drop=True).fillna(-1).to_pickle(f'../data/803_tmp{i}.p')


# =============================================================================
# colsample
# =============================================================================

# =============================================================================
# # train
# =============================================================================
load_folders = sorted(glob('../data/*_train/')) + ['../data/train/']

args = list(zip(load_folders, range(len(load_folders))))

pool = Pool(12)
pool.map(multi_train, args)
pool.close()

print('concat train')
load_files = sorted(glob('../data/803_tmp*.p'))
X = pd.concat([pd.read_pickle(f) for f in load_files], axis=1)
print('X.isnull().sum().sum():', X.isnull().sum().sum())

system('rm ../data/dtrain_501-3_top30.mt')

usecols_ = usecols[:30]
categorical_feature_ = list(set(categorical_feature) & set(usecols_))
lgb.Dataset(X[usecols_], label=X.is_attributed,
            categorical_feature=categorical_feature_).save_binary('../data/dtrain_501-3_top30.mt')
gc.collect()


system('rm ../data/dtrain_501-3_top50.mt')

usecols_ = usecols[:50]
categorical_feature_ = list(set(categorical_feature) & set(usecols_))
lgb.Dataset(X[usecols_], label=X.is_attributed,
            categorical_feature=categorical_feature_).save_binary('../data/dtrain_501-3_top50.mt')
gc.collect()

system('rm ../data/dtrain_501-3_top80.mt')

usecols_ = usecols[:80]
categorical_feature_ = list(set(categorical_feature) & set(usecols_))
lgb.Dataset(X[usecols_], label=X.is_attributed,
            categorical_feature=categorical_feature_).save_binary('../data/dtrain_501-3_top80.mt')
gc.collect()





X_head = X.head().drop('is_attributed', axis=1)
X_head.to_pickle('X_head.p')

del X; gc.collect()
system('rm ../data/803_tmp*.p')

"""

X_head = pd.read_pickle('X_head.p')

"""

# =============================================================================
# # test
# =============================================================================
sub = utils.read_pickles('../data/test_old', ['click_id'])

load_folders = sorted(glob('../data/*_test/')) + ['../data/test_old/']
args = list(zip(load_folders, range(len(load_folders))))

pool = Pool(15)
pool.map(multi_test, args)
pool.close()

print('concat test')
load_files = sorted(glob('../data/803_tmp*.p'))
X = pd.concat([pd.read_pickle(f) for f in load_files], axis=1)
print('test.shape should be 18790469:', X[X_head.columns].shape)
print('X.isnull().sum().sum():', X.isnull().sum().sum())

system('rm -rf ../data/dtest_501-3_top30')
usecols_ = usecols[:30]
utils.to_pickles(X[usecols_], '../data/dtest_501-3_top30', utils.SPLIT_SIZE)
gc.collect()

system('rm -rf ../data/dtest_501-3_top50')
usecols_ = usecols[:50]
utils.to_pickles(X[usecols_], '../data/dtest_501-3_top50', utils.SPLIT_SIZE)
gc.collect()

system('rm -rf ../data/dtest_501-3_top80')
usecols_ = usecols[:80]
utils.to_pickles(X[usecols_], '../data/dtest_501-3_top80', utils.SPLIT_SIZE)
gc.collect()


del X; gc.collect()


sub = sub[~sub.click_id.isnull()].reset_index(drop=True)
sub.drop_duplicates('click_id', keep='last', inplace=True) # last?
sub['click_id'] = sub['click_id'].map(int)
sub.reset_index(drop=True, inplace=True)

sub.to_pickle('../data/sub_501-3.p')
system('rm ../data/803_tmp*.p')

system('touch SUCCESS_803')

#==============================================================================
utils.end(__file__)




