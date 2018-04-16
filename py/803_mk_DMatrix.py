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
import xgboost as xgb
from multiprocessing import Pool
import utils
utils.start(__file__)

# setting
useimp = 100

system('rm ../data/tmp*.p')
#system('rm ../data/*.mt')

# =============================================================================
# imp
# =============================================================================
imp = pd.read_csv(glob('imp*.csv')[0])

imp.set_index('col', inplace=True)

imp = (imp - imp.mean()) / (imp.max() - imp.min())


imp['total'] = imp.sum(1)

imp.sort_values('total', ascending=False, inplace=True)

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
    df = pd.concat([pd.read_pickle(load_folder + '/{}.p'.format(j)) for j in [8,9]])
    col = list(set(df.columns) & usecols)
    if len(col)>0:
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
xgb.DMatrix(X.drop('is_attributed', axis=1), X.is_attributed).save_binary('../data/dtrain.mt')

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
xgb.DMatrix(X[X_head.columns]).save_binary('../data/dtest.mt')
del X; gc.collect()


sub = sub[~sub.click_id.isnull()].reset_index(drop=True)
sub.drop_duplicates('click_id', keep='last', inplace=True) # last?
sub['click_id'] = sub['click_id'].map(int)
sub.reset_index(drop=True, inplace=True)

sub.to_pickle('../data/sub.p')

system('touch SUCCESS')

#==============================================================================
utils.end(__file__)











