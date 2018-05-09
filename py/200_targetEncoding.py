#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 03:07:27 2018

@author: Kazuki
"""


import os
import pandas as pd
from tqdm import tqdm
import gc
from glob import glob
from multiprocessing import Pool
import utils
utils.start(__file__)


nthread = 10

os.system('rm -rf ../data/targetEncoding_*.p')

train = utils.read_pickles('../data/train')

def att(keys):
    print(keys)
    keys = list(keys)
    keys_ = '-'.join(keys)
    c = f'targetEncoding_{keys_}'
    
    gr = train.groupby(keys)
    sum_ = gr['is_attributed'].sum()
    gc.collect()
    size = gr.size()
    gc.collect()
    df = pd.concat([sum_, size], axis=1)
    df.columns = ['a', 'b']
    df[c] = df.a/df.b
    df['target_mean'] = df[c]
    df.to_pickle(f'../data/{c}.p')


# =============================================================================
# 
# =============================================================================
pool = Pool(nthread)
callback = pool.map(att, utils.comb)
pool.close()


#==============================================================================
utils.end(__file__)




