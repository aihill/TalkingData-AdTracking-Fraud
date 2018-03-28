#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:14:55 2018

@author: kazuki.onodera
"""

from glob import glob
import pandas as pd
import numpy as np
from os import system
from tqdm import tqdm
import gc
import xgboost as xgb
from multiprocessing import Pool
import threading
from queue import Queue
import utils
utils.start(__file__)

# setting
SEED = 71
useimp = 100
FRAC = 0.15
LOOP = 7
proc = 3

np.random.seed(SEED)

system('rm ../data/tmp*.p')
system('rm ../data/*.mt')

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

# =============================================================================
# def
# =============================================================================
def multi1(args):
    load_file, i = args
    gc.collect()
    print('loading {} ...'.format(load_file))
    df = pd.read_pickle(load_file)
    col = list(set(df.columns) & usecols)
    if len(col)>0:
        df[col].to_pickle('../data/tmp{}.p'.format(i))

df_queue = Queue()
#df_list = []
def sender(load_file):
    print('loading {} ...'.format(load_file))
    df_queue.put( pd.read_pickle(load_file) )
    print('loaded {}'.format(load_file))

# =============================================================================
# train colsample
# =============================================================================

train = pd.concat([utils.read_pickles('../data/train'),
                   pd.read_pickle('../data/101_train.p'),
                   pd.read_pickle('../data/102_train.p')], 
                  axis=1)#.sample(frac=0.4)

train[list(set(train.columns) & usecols)].to_pickle('../data/tmp{}.p'.format(0))

gc.collect()

load_files = ['../data/{}_train.p'.format('-'.join(k)) for k in utils.comb]
args = list(zip(load_files, range(1, len(load_files)+1 )))

pool = Pool(10)
pool.map(multi1, args)
pool.close()

gc.collect()

# read
# threading
threads = [None] * len(glob('../data/tmp*.p'))
for i, load_file in enumerate(glob('../data/tmp*.p')):
    threads[i] = threading.Thread(target=sender, args=(load_file, ))
    threads[i].start()

#df_queue.join()

for t in threads:
    t.join()


train = pd.concat([df_queue.get() for i in glob('../data/tmp*.p')], axis=1)
#train = pd.concat([pd.read_pickle(f) for f in glob('../data/tmp*.p')], axis=1)


y = train.is_attributed
train.drop( 'is_attributed', 
           axis=1, inplace=True)
train.fillna(-1, inplace=True)

gc.collect()

train_head = train.head()
train_head.to_pickle('train_head.p')

print('finish colsample')

# =============================================================================
# train
# =============================================================================

# valid
valid_seed = np.random.randint(99999)
X_valid = train.sample(frac=0.1, random_state=valid_seed)
y_valid = y.sample(frac=0.1, random_state=valid_seed)

dvalid = xgb.DMatrix(X_valid, y_valid)
dvalid.save_binary('../data/dvalid_10per.mt')

valid_index = X_valid.index
del dvalid, X_valid, y_valid; gc.collect()

# build 15per
build_seed = np.random.randint(99999)
X_build = train.drop(valid_index).sample(frac=0.15, random_state=build_seed)
y_build = y.drop(valid_index).sample(frac=0.15, random_state=build_seed)

xgb.DMatrix(X_build, y_build).save_binary('../data/dbuild_15per.mt')
gc.collect()

# build 10per
build_seed = np.random.randint(99999)
X_build = X_build.sample(frac=2/3, random_state=build_seed)
y_build = y_build.sample(frac=2/3, random_state=build_seed)

xgb.DMatrix(X_build, y_build).save_binary('../data/dbuild_10per.mt')
build_index = X_build.index
del X_build, y_build; gc.collect()



#train.drop(X_valid.index, inplace=True)
#y = y.drop(X_valid.index)



def multi2(argv):
    i, seed = argv
    print('saving {}...'.format(i))
    dtrain = xgb.DMatrix(train.sample(frac=FRAC, random_state=seed),
                         y.sample(frac=FRAC, random_state=seed))
    dtrain.save_binary('../data/dtrain{}.mt'.format(i))
    

cnt = 0
for i in range(LOOP):
    train_seeds = np.random.randint(99999, size=proc)
    pool = Pool(proc)
    argv = list(zip([cnt+j for j in range(proc)], train_seeds))
    pool.map(multi2, argv)
    pool.close()
    cnt += proc

system('rm ../data/tmp*.p')

del train, y; gc.collect()

# =============================================================================
# test colsample
# =============================================================================

test = pd.concat([utils.read_pickles('../data/test_old'),
                   pd.read_pickle('../data/101_test_old.p'),
                   pd.read_pickle('../data/102_test_old.p')], 
                  axis=1)#.sample(frac=0.4)

#target = ~test.click_id.isnull()
test[list(set(test.columns) & usecols)+['click_id']].to_pickle('../data/tmp{}.p'.format(0))

gc.collect()

load_files = ['../data/{}_test.p'.format('-'.join(k)) for k in utils.comb]
args = list(zip(load_files, range(1, len(load_files)+1 )))

pool = Pool(10)
pool.map(multi1, args)
pool.close()

gc.collect()

# read
# threading
threads = [None] * len(glob('../data/tmp*.p'))
for i, load_file in enumerate(glob('../data/tmp*.p')):
    threads[i] = threading.Thread(target=sender, args=(load_file, ))
    threads[i].start()

#df_queue.join()

for t in threads:
    t.join()


test = pd.concat([df_queue.get() for i in glob('../data/tmp*.p')], axis=1)

test = test[~test.click_id.isnull()]
test.drop_duplicates('click_id', keep='last', inplace=True) # last?

print('test.shape should be 18790469:', test.shape)

gc.collect()

sub = test[['click_id']].reset_index(drop=True)
sub.click_id = sub.click_id.map(int)

test.drop('click_id', axis=1, inplace=True)
test.fillna(-1, inplace=True)


xgb.DMatrix(test[train_head.columns]).save_binary('../data/dtest.mt')
sub.to_pickle('../data/sub.p')

system('rm ../data/tmp*.p')


#==============================================================================
utils.end(__file__)



