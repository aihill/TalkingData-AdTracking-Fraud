#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:44:57 2018

@author: Kazuki
"""

import pandas as pd
import numpy as np
from os import system
import os
from datetime import datetime
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
import lightgbm as lgb
import gc
from tqdm import tqdm
from multiprocessing import Pool
from glob import glob
import utils
utils.start(__file__)

SEED = 71 #np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999
is_sampling = True
FRAC = 0.7

train_files = [45, 46, 47, 48, 53, 54, 55, 56, 60, 61, 62, 63, 64, 65]
valid_files = [78, 79, 80, 81, 82, 88, 89, 90, 91, 95, 96, 97, 98]
print(f'train_files: {train_files}')
print(f'valid_files: {valid_files}')


np.random.seed(SEED)
print('seed :', SEED)


#system('rm ../data/801_tmp*.p')
system('rm SUCCESS_801')

categorical_feature = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']

# =============================================================================
# def
# =============================================================================

def multi_train_sampling(args):
    load_folder, i = args
    out_file = f'{load_folder[:-1]}_train_sampling.p'
    gc.collect()
    if os.path.isfile(out_file):
        print(f'{out_file} exist')
        return
    print(f'loading {load_folder} ...')
    
    if is_sampling==False:
        df = pd.concat([ pd.read_pickle(f'{load_folder}/{j:03d}.p') for j in train_files])
    else:
        df = pd.concat([ pd.read_pickle(f'{load_folder}/{j:03d}.p').sample(frac=FRAC, random_state=SEED) for j in train_files])
    
    print(f'writing {out_file} ...')
    df.reset_index(drop=True).fillna(-1).to_pickle(out_file)

def multi_valid_sampling(args):
    load_folder, i = args
    out_file = f'{load_folder[:-1]}_valid_sampling.p'
    gc.collect()
    if os.path.isfile(out_file):
        print(f'{out_file} exist')
        return
    print(f'loading {load_folder} ...')
    
    if is_sampling==False:
        df = pd.concat([ pd.read_pickle(f'{load_folder}/{j:03d}.p') for j in valid_files])
    else:
        df = pd.concat([ pd.read_pickle(f'{load_folder}/{j:03d}.p').sample(frac=FRAC, random_state=SEED) for j in valid_files])
        
    print(f'writing {out_file} ...')
    df.reset_index(drop=True).fillna(-1).to_pickle(out_file)

# =============================================================================
# load train
# =============================================================================    
load_folders = sorted(glob('../data/*train/'))

args = list(zip(load_folders, range(len(load_folders))))

pool = Pool(10)
pool.map(multi_train_sampling, args)
pool.close()


print('concat train')
load_files = sorted(glob('../data/*_train_sampling.p'))
X = pd.concat([pd.read_pickle(f) for f in tqdm(load_files)], axis=1)
print('X.isnull().sum().sum():', X.isnull().sum().sum())
drop_feature = ['click_time', 'attributed_time']
X.drop(drop_feature, axis=1, inplace=True)
X.fillna(-1, inplace=True)

print('X.shape:', X.shape )

system('rm ../data/dtrain.mt')

lgb.Dataset(X.drop('is_attributed', axis=1), label=X.is_attributed,
            categorical_feature=categorical_feature).save_binary('../data/dtrain.mt')

del X; gc.collect()

# =============================================================================
# load valid
# =============================================================================    
load_folders = sorted(glob('../data/*train/'))

args = list(zip(load_folders, range(len(load_folders))))

pool = Pool(10)
pool.map(multi_valid_sampling, args)
pool.close()


print('concat train')
load_files = sorted(glob('../data/*_valid_sampling.p'))
X = pd.concat([pd.read_pickle(f) for f in tqdm(load_files)], axis=1)
print('X.isnull().sum().sum():', X.isnull().sum().sum())
drop_feature = ['click_time', 'attributed_time']
X.drop(drop_feature, axis=1, inplace=True)
X.fillna(-1, inplace=True)

print('X.shape:', X.shape )

system('rm ../data/dvalid.mt')

lgb.Dataset(X.drop('is_attributed', axis=1), label=X.is_attributed,
            categorical_feature=categorical_feature).save_binary('../data/dvalid.mt')

del X; gc.collect()

#==============================================================================
utils.end(__file__)



