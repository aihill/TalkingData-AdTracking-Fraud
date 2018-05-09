#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 02:00:42 2018

@author: Kazuki
"""


#import numpy as np
import pandas as pd
#from tqdm import tqdm
import gc
import os
from glob import glob
from multiprocessing import Pool
nthread = 15
import utils
utils.start(__file__)


os.system('rm -rf ../data/702*')
os.system('mkdir ../data/702_train')
os.system('mkdir ../data/702_test')


def multi(ix):
    
    # train
    df = pd.concat([pd.read_pickle(f'../data/004_train/{ix:03d}.p'), 
                    pd.read_pickle(f'../data/005_train/{ix:03d}.p')], axis=1)
    
    col = []
    for keys in utils.comb:
        k1 = 'timedelta_' + '-'.join(keys)
        k2 = 'timedelta2_' + '-'.join(keys)
        k  = 'timeDD_'  + '-'.join(keys)
        df[k] = df[k1] - df[k2] # TODO: consider -1
        col.append(k)
        
        k1 = 'timedelta_rev_' + '-'.join(keys)
        k2 = 'timedelta2_rev_' + '-'.join(keys)
        k  = 'timeDD_rev_'  + '-'.join(keys)
        df[k] = df[k1] - df[k2]
        col.append(k)
    
    df[col].to_pickle(f'../data/702_train/{ix:03d}.p')
    
    del df; gc.collect()
    
    # test
    df = pd.concat([pd.read_pickle(f'../data/004_test/{ix:03d}.p'), 
                    pd.read_pickle(f'../data/005_test/{ix:03d}.p')], axis=1)
    
    col = []
    for keys in utils.comb:
        k1 = 'timedelta_' + '-'.join(keys)
        k2 = 'timedelta2_' + '-'.join(keys)
        k  = 'timeDD_'  + '-'.join(keys)
        df[k] = df[k1] - df[k2]
        col.append(k)
        
        k1 = 'timedelta_rev_' + '-'.join(keys)
        k2 = 'timedelta2_rev_' + '-'.join(keys)
        k  = 'timeDD_rev_'  + '-'.join(keys)
        df[k] = df[k1] - df[k2]
        col.append(k)
    
    df[col].to_pickle(f'../data/702_test/{ix:03d}.p')
    
    print(f'Finish {ix}')

# =============================================================================
# main
# =============================================================================

pool = Pool(nthread)
callback = pool.map(multi, range(utils.SPLIT_SIZE))
pool.close()



#==============================================================================
utils.end(__file__)
