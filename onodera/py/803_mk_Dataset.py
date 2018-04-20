#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:00:49 2018

@author: kazuki.onodera
"""

from glob import glob
import pandas as pd
from os import system
from tqdm import tqdm
import gc
import lightgbm as lgb
from multiprocessing import Pool
import utils
utils.start(__file__)

# setting
useimp = 50

system('rm ../data/tmp*.p')
system('rm ../data/*.mt')
system('rm SUCCESS_803')

categorical_feature = ['ip', 'app', 'device', 'os', 'channel']
# =============================================================================
# imp
# =============================================================================
filepath = sorted(glob('imp*.csv'))[-1]
print(filepath)
imp = pd.read_csv(filepath)

imp.set_index('index', inplace=True)


imp = imp.head(useimp).T

usecols = set(imp.columns.tolist() + ['is_attributed'])
print(usecols)

# =============================================================================
# def
# =============================================================================

def multi_train(args):
    load_folder, i = args
    gc.collect()
    print('loading {} ...'.format(load_folder))
    df = pd.read_pickle(load_folder + '/0.p')
    col = list(set(df.columns) & usecols)
    df = pd.concat([pd.read_pickle(load_folder + '/{}.p'.format(j))[col] for j in range(0, 10)])
    if len(col)>0:
        gc.collect()
        df[col].to_pickle('../data/tmp{}.p'.format(i))

def multi_test(args):
    load_folder, i = args
    gc.collect()
    print('loading {} ...'.format(load_folder))
    df = pd.concat([sub, utils.read_pickles(load_folder)],
                    axis=1)
    col = list(set(df.columns) & usecols)
    if len(col)>0:
        df = df[~df.click_id.isnull()]
        df.drop_duplicates('click_id', keep='last', inplace=True) # last?
        print(load_folder, df.shape)
        df[col].reset_index(drop=True).to_pickle('../data/tmp{}.p'.format(i))


# =============================================================================
# colsample
# =============================================================================

# =============================================================================
# # train
# =============================================================================
load_folders = sorted(glob('../data/*train/'))

args = list(zip(load_folders, range(len(load_folders))))

pool = Pool(10)
pool.map(multi_train, args)
pool.close()

print('concat train')
load_files = sorted(glob('../data/tmp*.p'))
X = pd.concat([pd.read_pickle(f) for f in load_files], axis=1)
lgb.Dataset(X.drop('is_attributed', axis=1), label=X.is_attributed,
            categorical_feature=categorical_feature).save_binary('../data/dtrain.mt')

X_head = X.head().drop('is_attributed', axis=1)
X_head.to_pickle('X_head.p')

del X; gc.collect()
system('rm ../data/tmp*.p')

"""

X_head = pd.read_pickle('X_head.p')

"""

# =============================================================================
# # test
# =============================================================================
sub = utils.read_pickles('../data/test_old', ['click_id'])

load_folders = sorted(glob('../data/*_test/'))
args = list(zip(load_folders, range(len(load_folders))))

pool = Pool(5)
pool.map(multi_test, args)
pool.close()

print('concat test')
load_files = sorted(glob('../data/tmp*.p'))
X = pd.concat([pd.read_pickle(f) for f in load_files], axis=1)
print('test.shape should be 18790469:', X[X_head.columns].shape)
lgb.Dataset(X[X_head.columns], 
            categorical_feature=categorical_feature).save_binary('../data/dtest.mt')
del X; gc.collect()


sub = sub[~sub.click_id.isnull()].reset_index(drop=True)
sub.drop_duplicates('click_id', keep='last', inplace=True) # last?
sub['click_id'] = sub['click_id'].map(int)
sub.reset_index(drop=True, inplace=True)

sub.to_pickle('../data/sub.p')

system('touch SUCCESS_803')

#==============================================================================
utils.end(__file__)











