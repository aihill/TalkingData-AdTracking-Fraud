#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:24:03 2018

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
usecols = list(set(utils.BEST_FEATURES_502 + utils.BEST_FEATURES_504))
usecols_set = set(usecols)
print(sorted(usecols))

system('rm ../data/805_tmp*.p')
#system('rm ../data/*.mt')
system('rm SUCCESS_805')

categorical_feature = list( set(['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']) & usecols_set)

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
        print(f'writing ../data/805_tmp{i}.p ...')
        df[col].reset_index(drop=True).fillna(-1).to_pickle(f'../data/805_tmp{i}.p')

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
        df[col].reset_index(drop=True).fillna(-1).to_pickle(f'../data/805_tmp{i}.p')


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
load_files = sorted(glob('../data/805_tmp*.p'))
X = pd.concat([pd.read_pickle(f) for f in load_files], axis=1)
print('X.isnull().sum().sum():', X.isnull().sum().sum())

system('rm ../data/dtrain.mt')
system('rm -rf ../data/dtrain')

y = utils.read_pickles('../data/is_attributed')
lgb.Dataset(X, label=y,
            categorical_feature=categorical_feature).save_binary('../data/dtrain.mt')
utils.to_pickles(X, '../data/dtrain', utils.SPLIT_SIZE)


X_head = X.head()
X_head.to_pickle('X_head.p')

del X, y; gc.collect()
system('rm ../data/805_tmp*.p')

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
load_files = sorted(glob('../data/805_tmp*.p'))
X = pd.concat([pd.read_pickle(f) for f in load_files], axis=1)
print('test.shape should be 18790469:', X[X_head.columns].shape)
print('X.isnull().sum().sum():', X.isnull().sum().sum())

system('rm -rf ../data/dtest')
utils.to_pickles(X[X_head.columns], '../data/dtest', utils.SPLIT_SIZE)

del X; gc.collect()


sub = sub[~sub.click_id.isnull()].reset_index(drop=True)
sub.drop_duplicates('click_id', keep='last', inplace=True) # last?
sub['click_id'] = sub['click_id'].map(int)
sub.reset_index(drop=True, inplace=True)

sub.to_pickle('../data/sub.p')
system('rm ../data/805_tmp*.p')

system('touch SUCCESS_805')

#==============================================================================
utils.end(__file__)




