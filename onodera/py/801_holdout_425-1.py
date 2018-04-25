#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:27:08 2018

@author: kazuki.onodera
"""

import pandas as pd
import numpy as np
from os import system
import os
from datetime import datetime
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
#import lightgbm as lgb
import gc
#from time import sleep
from multiprocessing import Pool
from glob import glob
import utils
utils.start(__file__)

SEED = 71 #np.random.randint(9999) #int(sys.argv[1])
NROUND = 9999
FRAC = 0.1
LOAD_SIZE = 10

np.random.seed(SEED)
print('seed :', SEED)


#system('rm ../data/801_tmp*.p')
system('rm SUCCESS_801')

utils.send_line('START {}'.format(__file__))

# =============================================================================
# def
# =============================================================================
load_files = list(np.random.choice(range(utils.SPLIT_SIZE), replace=False, size=LOAD_SIZE))
print(f'load_files: {load_files}')

def multi_train(args):
    load_folder, i = args
    out_file = f'{load_folder[:-1]}_sampled.p'
    gc.collect()
    if os.path.isfile(out_file):
        print(f'{out_file} exist')
        return
    print(f'loading {load_folder} ...')
    df = pd.concat([ pd.read_pickle(f'{load_folder}/{j:03d}.p') for j in load_files])
#    df = utils.read_pickles(load_folder).sample(frac=FRAC, random_state=SEED)
    print(f'writing {out_file} ...')
    df.reset_index(drop=True).fillna(-1).to_pickle(out_file)

# =============================================================================
# load train
# =============================================================================    
load_folders = sorted(glob('../data/*train/'))

args = list(zip(load_folders, range(len(load_folders))))

pool = Pool(10)
pool.map(multi_train, args)
pool.close()


print('concat train')
load_files = sorted(glob('../data/*_sampled.p'))
X = pd.concat([pd.read_pickle(f) for f in load_files], axis=1).sample(frac=0.5, random_state=SEED)
print('X.isnull().sum().sum():', X.isnull().sum().sum())
print('X.shape:', X.shape )

gc.collect()

y = X.is_attributed
categorical_feature = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']

drop_feature = ['is_attributed', 'click_time', 'attributed_time']
X.drop(drop_feature, axis=1, inplace=True)
X.fillna(-1, inplace=True)

gc.collect()

col = X.columns
print(col)

# =============================================================================
# lgbm
# =============================================================================

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.2,
         'max_depth': 4,
         'num_leaves': 2**4-1,
         'max_bin': 100,
         'min_child_samples': 300,
         'min_child_weight': 0,
         'colsample_bytree': 0.8,
         'subsample': 0.1,
         'nthread': 64,
         'scale_pos_weight': 500,
         
         'seed': SEED
         }

gc.collect()
yhat, imp, ret = ex.stacking(X, y, param, NROUND, nfold=5, esr=50, 
                             categorical_feature=categorical_feature)

t = datetime.today()
date = t.date()
hour = t.hour
imp.to_csv('imp_{}-{:02d}h.csv'.format(date, hour), index=False)



system('touch SUCCESS_801')



#==============================================================================
utils.end(__file__)
