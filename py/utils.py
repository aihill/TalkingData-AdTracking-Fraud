
"""

base columns
['ip', 'app', 'device', 'os', 'channel', 'click_time', 
'attributed_time', 'is_attributed']

"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
import time
import gc

# =============================================================================
# global variables
# =============================================================================
sort_keys = ['ip', 'app', 'device', 'os', 'channel', 'click_time']



# =============================================================================
# def
# =============================================================================
def start(fname):
    global st_time
    st_time = time.time()
    print("""
#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(fname, os.getpid()))
    
    return

def end(fname):
    
    print("""
#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(fname))
    print('time: {:.2f}min'.format( (time.time() - st_time)/60 ))
    return

def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)
    
def to_pickles(df, path, split_size=3, inplace=True):
    """
    path = '../output/mydf'
    
    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'
    
    """
    if inplace==True:
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.reset_index(drop=True)
    gc.collect()
    mkdir_p(path)
    
#    for i in tqdm(range(split_size)):
#        df.ix[df.index%split_size==i].to_pickle(path+'/{}.p'.format(i))
    
    kf = KFold(n_splits=split_size)
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
        df.iloc[val_index].to_pickle(path+'/{}.p'.format(i))
    return

def read_pickles(path, col=None):
    if col is None:
        df = pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*')))])
    else:
        df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
    return df

def submit(file_path):
    os.system('kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f {} -m "from API"'.format(file_path))



