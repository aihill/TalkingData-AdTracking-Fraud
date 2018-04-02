#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 02:23:57 2018

@author: Kazuki

takes 90 minutes

"""


from glob import glob
from os import system
import pandas as pd
from tqdm import tqdm
import gc
from multiprocessing import Pool
import utils
utils.start(__file__)

# =============================================================================
# load
# =============================================================================

train = utils.read_pickles('../data/train').drop(['attributed_time', 'is_attributed'], axis=1)
test = utils.read_pickles('../data/test_old').drop(['click_id'], axis=1)


train['day'] = train.click_time.dt.day
train['hour'] = train.click_time.dt.hour

test['day'] = test.click_time.dt.day
test['hour'] = test.click_time.dt.hour
gc.collect()

# =============================================================================
# features
# =============================================================================

# std
def multi(keys):
    
    gc.collect()
    keys_ = '-'.join(keys)
    print(keys)
    
    # for train
    df = train.groupby(list(keys) + ['day', 'hour']).size()
    df.name = keys_+'_dayhour_count'
    df = df.reset_index()
    
    train_ = pd.merge(train, df, on=keys, how='left')
    train_[[keys_+'_dayhour_count']].to_pickle('../data/105_train_{}_dayhour_count.p'.format(keys_))
    del train_, df; gc.collect()
    
    # for test
    df = test.groupby(list(keys) + ['day', 'hour']).size()
    df.name = keys_+'_dayhour_count'
    df = df.reset_index()
    
    test_ = pd.merge(test, df, on=keys, how='left')
    test_[[keys_+'_dayhour_count']].to_pickle('../data/105_test_{}_dayhour_count.p'.format(keys_))



pool = Pool(3)
comb = [c for c in utils.comb if len(c)<=2]
callback = pool.map(multi, comb)
pool.close()

# straight
[multi(keys) for keys in utils.comb if len(keys)<=2]

# =============================================================================
# concat
# =============================================================================


print('concat train')
pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob('../data/105_train_*.p')))], axis=1).to_pickle('../data/105_train.p')
system('rm ../data/105_train_*')

gc.collect()

print('concat test')
pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob('../data/105_test_*.p')))], axis=1).to_pickle('../data/105_test.p')
system('rm ../data/105_test_*')


#==============================================================================
utils.end(__file__)
