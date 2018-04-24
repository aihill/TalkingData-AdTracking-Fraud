#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:00:49 2018

@author: kazuki.onodera
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
useimp = 60
filepath = 'imp_2018-04-24-00h.csv'



system('rm ../data/802_tmp*.p')
#system('rm ../data/*.mt')
system('rm SUCCESS_802')

categorical_feature = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']
categorical_feature += ['preChannel_app-device-os', 'nextChannel_app-device-os',
                       'preChannel_app-device', 'nextChannel_app-device', 'preChannel_app-os',
                       'nextChannel_app-os', 'preChannel_app', 'nextChannel_app',
                       'preApp_channel', 'nextApp_channel', 'preApp_device-channel',
                       'nextApp_device-channel', 'preApp_device-os-channel',
                       'nextApp_device-os-channel', 'preApp_device-os', 'nextApp_device-os',
                       'preChannel_device-os', 'nextChannel_device-os', 'preApp_device',
                       'nextApp_device', 'preChannel_device', 'nextChannel_device',
                       'preChannel_ip-app-device-os', 'nextChannel_ip-app-device-os',
                       'preChannel_ip-app-device', 'nextChannel_ip-app-device',
                       'preChannel_ip-app-os', 'nextChannel_ip-app-os', 'preChannel_ip-app',
                       'nextChannel_ip-app', 'preApp_ip-channel', 'nextApp_ip-channel',
                       'preApp_ip-device-channel', 'nextApp_ip-device-channel',
                       'preApp_ip-device-os-channel', 'nextApp_ip-device-os-channel',
                       'preApp_ip-device-os', 'nextApp_ip-device-os',
                       'preChannel_ip-device-os', 'nextChannel_ip-device-os',
                       'preApp_ip-device', 'nextApp_ip-device', 'preChannel_ip-device',
                       'nextChannel_ip-device', 'preApp_ip-os-channel',
                       'nextApp_ip-os-channel', 'preApp_ip-os', 'nextApp_ip-os',
                       'preChannel_ip-os', 'nextChannel_ip-os', 'preApp_ip', 'nextApp_ip',
                       'preChannel_ip', 'nextChannel_ip', 'preApp_os-channel',
                       'nextApp_os-channel', 'preApp_os', 'nextApp_os', 'preChannel_os',
                       'nextChannel_os']

# =============================================================================
# wait
# =============================================================================
while True:
    if os.path.isfile('SUCCESS_801'):
        break
    else:
        sleep(60*1)

utils.send_line('START {}'.format(__file__))

# =============================================================================
# imp
# =============================================================================
print(filepath)
imp = pd.read_csv(filepath)

imp.set_index('index', inplace=True)


imp = imp.head(useimp).T

usecols = set(imp.columns.tolist() + ['is_attributed'])
categorical_feature = list(set(categorical_feature) & usecols)

print(sorted(usecols))

# =============================================================================
# def
# =============================================================================

def multi_train(args):
    load_folder, i = args
    gc.collect()
    print('loading {} ...'.format(load_folder))
    df = pd.read_pickle(load_folder + '/0.p')
    col = list(set(df.columns) & usecols)
    if len(col)>0:
        df = pd.concat([pd.read_pickle(load_folder + '/{}.p'.format(j))[col] for j in range(0, 10)])
        gc.collect()
        df[col].reset_index(drop=True).fillna(-1).to_pickle(f'../data/802_tmp{i}.p')

def multi_test(args):
    load_folder, i = args
    gc.collect()
    print('loading {} ...'.format(load_folder))
    if load_folder=='../data/test_old/':
        df = utils.read_pickles(load_folder)
        
    else:
        df = pd.read_pickle(load_folder + '/0.p')
        col = list(set(df.columns) & usecols)
        if len(col)==0:
            return
        df = pd.concat([sub, utils.read_pickles(load_folder)],
                        axis=1)
    col = list(set(df.columns) & usecols)
    if len(col)>0:
        df = df[~df.click_id.isnull()]
        df.drop_duplicates('click_id', keep='last', inplace=True) # last?
        print(load_folder, df.shape)
        df[col].reset_index(drop=True).fillna(-1).to_pickle(f'../data/802_tmp{i}.p')


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
load_files = sorted(glob('../data/802_tmp*.p'))
X = pd.concat([pd.read_pickle(f) for f in load_files], axis=1)
print('X.isnull().sum().sum():', X.isnull().sum().sum())

system('rm ../data/dtrain.mt')

lgb.Dataset(X.drop('is_attributed', axis=1), label=X.is_attributed,
            categorical_feature=categorical_feature).save_binary('../data/dtrain.mt')

X_head = X.head().drop('is_attributed', axis=1)
X_head.to_pickle('X_head.p')

del X; gc.collect()
system('rm ../data/802_tmp*.p')

"""

X_head = pd.read_pickle('X_head.p')

"""

# =============================================================================
# # test
# =============================================================================
sub = utils.read_pickles('../data/test_old', ['click_id'])

load_folders = sorted(glob('../data/*_test/')) + ['../data/test_old/']
args = list(zip(load_folders, range(len(load_folders))))

pool = Pool(14)
pool.map(multi_test, args)
pool.close()

print('concat test')
load_files = sorted(glob('../data/802_tmp*.p'))
X = pd.concat([pd.read_pickle(f) for f in load_files], axis=1)
print('test.shape should be 18790469:', X[X_head.columns].shape)
print('X.isnull().sum().sum():', X.isnull().sum().sum())

utils.to_pickles(X[X_head.columns], '../data/dtest', 10)

del X; gc.collect()


sub = sub[~sub.click_id.isnull()].reset_index(drop=True)
sub.drop_duplicates('click_id', keep='last', inplace=True) # last?
sub['click_id'] = sub['click_id'].map(int)
sub.reset_index(drop=True, inplace=True)

sub.to_pickle('../data/sub.p')
system('rm ../data/802_tmp*.p')

system('touch SUCCESS_802')

#==============================================================================
utils.end(__file__)











