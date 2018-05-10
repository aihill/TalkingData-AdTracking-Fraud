#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:05:04 2018

@author: Kazuki
"""


#import numpy as np
import pandas as pd
#from tqdm import tqdm
import gc
import os
from glob import glob
from multiprocessing import Pool
nthread = 12
#from collections import defaultdict
import utils
utils.start(__file__)



def multi(ix):
    
    # train
    df = pd.read_pickle(f'../data/008_train/{ix:03d}.p')
    col = [c.replace('timedelta_', 'timedeltaV2_') for c in df.columns]
    col_di = dict(zip(df.columns, col))
    
    df.rename(columns=col_di).to_pickle(f'../data/008_train/{ix:03d}.p')
    
    del df; gc.collect()
    
    # test
    df = pd.read_pickle(f'../data/008_test/{ix:03d}.p')
    
    df.rename(columns=col_di).to_pickle(f'../data/008_test/{ix:03d}.p')
    
    print(f'Finish {ix}')

# =============================================================================
# main
# =============================================================================

pool = Pool(nthread)
callback = pool.map(multi, range(utils.SPLIT_SIZE))
pool.close()



#==============================================================================
utils.end(__file__)
