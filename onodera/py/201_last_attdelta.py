#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 15:01:40 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
from time import time
import gc
import os
from glob import glob
from multiprocessing import Pool
nthread = 5
#from collections import defaultdict
import utils
utils.start(__file__)


os.system('rm -rf ../data/201__*.p')

trte = pd.concat([utils.read_pickles('../data/train'),
                utils.read_pickles('../data/test_old')],
                ignore_index=True)[['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']]

gc.collect()


def multi(count_keys):
    """
    ex:
    count_keys = ('ip', 'app', 'device')
    
    """
    st = time()
    gc.collect()
    print(count_keys)
    count_keys = list(count_keys)
    
    count_keys_ = '-'.join(count_keys)
    keys = count_keys+['click_time']
    
    # plan1
    df = trte[keys+['attributed_time']].sort_values(keys)
    
    # plan2
#    df = trte[keys+['attributed_time']]
#    df = df[df.duplicated(count_keys, False)].sort_values(keys)
    
    gc.collect()
    
    c1 = 'pre_att_time_'+count_keys_
    df[c1] = df.groupby(count_keys).attributed_time.fillna(method='ffill').shift()
    df['key_match'] = ( df[count_keys]==df[count_keys].shift() ).all(1)*1
    df.loc[df.key_match==0, c1] = np.nan
    df[c1] = (df.click_time - df[c1]).dt.seconds
    df.loc[df[c1]<0, c1] = np.nan # TODO: necessary?
    gc.collect()
    
    df.drop(count_keys, axis=1, inplace=True)
    df.sort_index(inplace=True)
    df.fillna(-1, inplace=True)
    gc.collect()
    
    df.iloc[0:utils.TRAIN_SHAPE][[c1]].to_pickle('../data/201__{}_train.p'.format(count_keys_))
    df.iloc[utils.TRAIN_SHAPE:][[c1]].to_pickle('../data/201__{}_test.p'.format(count_keys_))
    
    print(f'Finished {count_keys} {(time()-st)/60:.3f} min')
    


pool = Pool(nthread)
callback = pool.map(multi, utils.comb)
pool.close()

del trte; gc.collect()

# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/201__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/201_train', 10)

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/201__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/201_test', 10)

os.system('rm -rf ../data/201__*.p')



#==============================================================================
utils.end(__file__)

