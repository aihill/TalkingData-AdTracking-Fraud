#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:43:36 2018

@author: kazuki.onodera
"""


#import numpy as np
import pandas as pd
#from tqdm import tqdm
import gc
import os
from glob import glob
#from collections import defaultdict
import utils
utils.start(__file__)

# =============================================================================
# concat
# =============================================================================

# train
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/004__*_train.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/004_train', utils.SPLIT_SIZE)

del df; gc.collect()

# test
df = pd.concat([pd.read_pickle(f) for f in sorted(glob('../data/004__*_test.p'))], axis=1).reset_index(drop=True)
utils.to_pickles(df, '../data/004_test', utils.SPLIT_SIZE)

os.system('rm -rf ../data/004__*.p')



#==============================================================================
utils.end(__file__)

