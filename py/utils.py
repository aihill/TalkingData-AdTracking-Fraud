
"""

base columns
['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']

"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm
from itertools import combinations
from sklearn.model_selection import KFold
from time import time
from datetime import datetime
import gc

# =============================================================================
# global variables
# =============================================================================
comb = []
for i in range(1, 6):
    comb += list(combinations(['ip', 'app', 'device', 'os', 'channel'], i))

#sort_keys = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
sort_keys = ['click_time']


# =============================================================================
# def
# =============================================================================
def start(fname):
    global st_time
    st_time = time()
    print("""
#==============================================================================
# START!!! {}    PID: {}    time: {}
#==============================================================================
""".format( fname, os.getpid(), datetime.today() ))
    
    return

def end(fname):
    print("""
#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(fname))
    print('time: {:.2f}min'.format( elapsed_minute() ))
    return

def elapsed_minute():
    return (time() - st_time)/60


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
        df.iloc[val_index].to_pickle(path+'/{}.p.gz'.format(i), compression='gzip')
    return

def read_pickles(path, col=None):
    if col is None:
        df = pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*')))])
    else:
        df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
    return df

def reduce_memory(df, ix_start=0):
    df.fillna(-1, inplace=True)
    if df.shape[0]>9999:
        df_ = df.sample(9999, random_state=71)
    else:
        df_ = df
    ## int
    col_int8 = []
    col_int16 = []
    col_int32 = []
#    for c in tqdm(df.columns[ix_start:], miniters=20):
    for c in df.columns[ix_start:]:
        if df[c].dtype=='O':
            continue
        if (df_[c] == df_[c].astype(np.int8)).all():
            col_int8.append(c)
        elif (df_[c] == df_[c].astype(np.int16)).all():
            col_int16.append(c)
        elif (df_[c] == df_[c].astype(np.int32)).all():
            col_int32.append(c)
    
    df[col_int8]  = df[col_int8].astype(np.int8)
    df[col_int16] = df[col_int16].astype(np.int16)
    df[col_int32] = df[col_int32].astype(np.int32)
    
    ## float
    col = [c for c in df.dtypes[df.dtypes==np.float64].index if '_id' not in c]
    df[col] = df[col].astype(np.float32)

    gc.collect()


def submit(file_path):
    os.system('kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f {} -m "from API"'.format(file_path))



