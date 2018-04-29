#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 13:46:34 2018

@author: Kazuki
"""


import pandas as pd
import numpy as np
from os import system
import os
from datetime import datetime
import sys
#sys.path.append('/home/kazuki_onodera/Python')
#import lgbmextension as ex
#import lightgbm as lgb
import gc
from tqdm import tqdm
from multiprocessing import Pool
from glob import glob
import utils
utils.start(__file__)
# =============================================================================
NTHREAD = 16

SEED = 71 #np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999
DO_SAMPLING = True
FRAC = 0.7

DO_CONCAT = False

# =============================================================================
np.random.seed(SEED)
print('seed :', SEED)

#system('rm ../data/*sampling.f')
system('rm SUCCESS_801')

train_files = [45, 46, 47, 48, 53, 54, 55, 56, 60, 61, 62, 63, 64, 65]
valid_files = [78, 79, 80, 81, 82, 88, 89, 90, 91, 95, 96, 97, 98]
print(f'train_files: {train_files}')
print(f'valid_files: {valid_files}')

categorical_feature = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']

# =============================================================================
# def
# =============================================================================

def multi_train_sampling(args):
    load_folder, i = args
    out_file = f'{load_folder[:-1]}_train_sampling.f'
    gc.collect()
    if os.path.isfile(out_file):
        print(f'{out_file} exist')
        return
    print(f'loading {load_folder} ...')
    
    if DO_SAMPLING==False:
        df = pd.concat([ pd.read_pickle(f'{load_folder}/{j:03d}.p') for j in train_files])
    else:
        df = pd.concat([ pd.read_pickle(f'{load_folder}/{j:03d}.p').sample(frac=FRAC, random_state=SEED) for j in train_files])
    
    print(f'writing {out_file} ...')
    df.reset_index(drop=True).fillna(-1).to_feather(out_file)

def multi_valid_sampling(args):
    load_folder, i = args
    out_file = f'{load_folder[:-1]}_valid_sampling.f'
    gc.collect()
    if os.path.isfile(out_file):
        print(f'{out_file} exist')
        return
    print(f'loading {load_folder} ...')
    
    if DO_SAMPLING==False:
        df = pd.concat([ pd.read_pickle(f'{load_folder}/{j:03d}.p') for j in valid_files])
    else:
        df = pd.concat([ pd.read_pickle(f'{load_folder}/{j:03d}.p').sample(frac=FRAC, random_state=SEED) for j in valid_files])
        
    print(f'writing {out_file} ...')
    df.reset_index(drop=True).fillna(-1).to_feather(out_file)

# =============================================================================
# load train
# =============================================================================    
load_folders = sorted(glob('../data/*train/'))

args = list(zip(load_folders, range(len(load_folders))))

pool = Pool(NTHREAD)
pool.map(multi_train_sampling, args)
pool.close()

if DO_CONCAT:
    print('concat train')
    load_files = sorted(glob('../data/*_train_sampling.f'))
    X = pd.concat([pd.read_feather(f) for f in tqdm(load_files)], axis=1)
    print('X.isnull().sum().sum():', X.isnull().sum().sum())
    drop_feature = ['click_time', 'attributed_time']
    X.drop(drop_feature, axis=1, inplace=True)
    X.fillna(-1, inplace=True)
    
    print('X.shape:', X.shape )
    
    
    y = X[['is_attributed']]
    X.drop('is_attributed', axis=1, inplace=True); gc.collect()
    
    #lgb.Dataset(X, label=y,
    #            categorical_feature=categorical_feature).save_binary('../data/dtrain.mt')
    
    X.to_feather('../data/X_train.f')
    y.to_feather('../data/y_train.f')
    
    X_head = X.head()
    X_head.to_pickle('X_head.p')
    
    """
    
    X_head = pd.read_pickle('X_head.p')
    
    """

    del X; gc.collect()

# =============================================================================
# load valid
# =============================================================================    
load_folders = sorted(glob('../data/*train/'))

args = list(zip(load_folders, range(len(load_folders))))

pool = Pool(NTHREAD)
pool.map(multi_valid_sampling, args)
pool.close()

if DO_CONCAT:
    print('concat valid')
    load_files = sorted(glob('../data/*_valid_sampling.f'))
    X = pd.concat([pd.read_feather(f) for f in tqdm(load_files)], axis=1)
    print('X.isnull().sum().sum():', X.isnull().sum().sum())
    drop_feature = ['click_time', 'attributed_time']
    X.drop(drop_feature, axis=1, inplace=True)
    X.fillna(-1, inplace=True)
    
    print('X.shape:', X.shape )
    
    
    y = X[['is_attributed']]
    X.drop('is_attributed', axis=1, inplace=True); gc.collect()
    
    #lgb.Dataset(X[X_head.columns], label=y,
    #            categorical_feature=categorical_feature).save_binary('../data/dvalid.mt')
    
    X[X_head.columns].to_feather('../data/X_valid.f')
    y.to_feather('../data/y_valid.f')
    
    del X; gc.collect()
    
    system('touch SUCCESS_801')

#==============================================================================
utils.end(__file__)



