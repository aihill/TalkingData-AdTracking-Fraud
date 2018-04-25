#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 23:47:38 2018

@author: kazuki.onodera
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


os.system('rm -rf ../data/701*')
os.system('mkdir ../data/701_train')
os.system('mkdir ../data/701_test')


def multi(ix):
    
    # train
    df = pd.concat([pd.read_pickle(f'../data/002_train/{ix:03d}.p'), 
                    pd.read_pickle(f'../data/101_train/{ix:03d}.p')], axis=1)
    
    col = []
    for keys in utils.comb:
        k1 = 'count_' + '-'.join(keys)
        k2 = 'totalcount_' + '-'.join(keys)
        k  = 'countratio_'  + '-'.join(keys)
        df[k] = df[k1]/df[k2]
        col.append(k)
    
    df[col].to_pickle(f'../data/701_train/{ix:03d}.p')
    
    del df; gc.collect()
    
    # test
    df = pd.concat([pd.read_pickle(f'../data/002_test/{ix:03d}.p'), 
                    pd.read_pickle(f'../data/101_test/{ix:03d}.p')], axis=1)
    
    col = []
    for keys in utils.comb:
        k1 = 'count_' + '-'.join(keys)
        k2 = 'totalcount_' + '-'.join(keys)
        k  = 'countratio_'  + '-'.join(keys)
        df[k] = df[k1]/df[k2]
        col.append(k)
    
    df[col].to_pickle(f'../data/701_test/{ix:03d}.p')
    
    print(f'Finish {ix}')

# =============================================================================
# main
# =============================================================================

pool = Pool(nthread)
callback = pool.map(multi, range(utils.SPLIT_SIZE))
pool.close()



#==============================================================================
utils.end(__file__)
