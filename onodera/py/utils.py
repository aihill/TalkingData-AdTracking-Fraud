
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

#TRAIN_SHAPE = 183698397
TRAIN_SHAPE = 184903890

SPLIT_SIZE = 100

# 429-2
BEST_FEATURES_429 = ['app', 'channel', 'count_app-device', 'count_device', 
                 'count_device-channel', 'count_ip', 'count_ip-app-device', 
                 'count_ip-app-device-os', 'count_ip-device', 'count_ip-device-os', 
                 'countratio_app', 'countratio_app-device', 'countratio_app-os', 
                 'countratio_channel', 'countratio_device', 'countratio_os', 
                 'dayvar_app-os', 'dayvar_device-os', 'device', 'hour', 'hour_min',
                 'hourvar_ip', 'is_attributed', 'nunique_app-device-os-channel_app-channel',
                 'nunique_device-channel_channel', 'nunique_ip-app-channel_app-channel', 
                 'nunique_ip-app-device-channel_ip-device', 'nunique_ip-app-device-os_app-device', 
                 'nunique_ip-app-device-os_ip-os', 'nunique_ip-app-device_ip', 
                 'nunique_ip-app-device_ip-device', 'nunique_ip-app-os_app', 
                 'nunique_ip-app-os_ip', 'nunique_ip-app-os_ip-os', 'nunique_ip-app_ip',
                 'nunique_ip-device-channel_ip', 'nunique_ip-device-channel_ip-channel', 
                 'nunique_ip-device-os-channel_ip-os', 'nunique_ip-device-os_ip-device', 
                 'nunique_ip-device-os_ip-os', 'nunique_ip-device_ip', 'nunique_ip-os_ip', 
                 'os', 'sameClickTimeCount_app-channel', 'timeDD_rev_ip-app', 'timeDD_rev_ip-app-device', 
                 'timeDD_rev_ip-app-device-os', 'timeDD_rev_ip-app-os', 'timeDD_rev_ip-device', 
                 'timedelta2_rev_ip', 'timedelta2_rev_ip-app-device-os', 'timedelta2_rev_ip-app-os',
                 'timedelta2_rev_ip-app-os-channel', 'timedelta2_rev_ip-device', 
                 'timedelta2_rev_ip-device-os-channel', 'timedelta_rev_ip-app', 
                 'timedelta_rev_ip-app-device', 'timedelta_rev_ip-app-device-os', 
                 'timedelta_rev_ip-app-device-os-channel', 'timedelta_rev_ip-app-os', 
                 'timedelta_rev_ip-app-os-channel', 'timedelta_rev_ip-device', 
                 'timedelta_rev_ip-device-os', 'timedelta_rev_ip-os', 'timediff-meadian_app', 
                 'timediff-meadian_channel', 'timediff-minmax_app-channel', 'timeskew_ip', 
                 'totalCountByDay_app', 'totalCountByDay_app-channel', 'totalCountByDay_app-device-channel', 
                 'totalCountByDay_app-device-os', 'totalCountByDay_ip', 'totalCountByDay_ip-app', 
                 'totalRatioByDay_ip', 'totalcount_app', 'totalcount_app-device', 'totalcount_app-device-os',
                 'totalcount_device-os', 'totalcount_ip-device', 'totalcount_ip-device-os']

BEST_FEATURES_502 = ['app', 'os', 'timedelta_rev_ip-app-device-os', 'hour_min', 'device', 
           'timeDD_rev_ip-app-device-os', 'nunique_ip-app_ip', 'nunique_ip-app-device-os_ip-os', 
           'nunique_ip-os_ip', 'timediff-minmax_app-channel', 'totalcount_app', 
           'nunique_app-device-os-channel_app-channel', 'timeskew_ip', 
           'nunique_device-os-channel_channel', 'totalcount_ip-app', 
           'nunique_ip-app-device-channel_ip-channel', 'timevar_app-os-channel', 
           'timedelta2_app-device-os-channel', 'timediff-meadian_ip-app-device-channel',
           'timeDD_device-os', 'timeDD_rev_ip-app-device-os-channel', 'timedelta2_rev_ip-app-os', 
           'nunique_ip-app-device_ip-device', 'totalcount_ip-device', 'timedelta_ip',
           'nunique_device-channel_channel', 'totalCountByDay_ip-app-device', 
           'sameClickTimeCount_device-os', 'timeskew_app-device', 'nunique_ip-app-device-channel_ip-app', 
           'nunique_ip-device-os_os', 'totalCountByDay_app-os-channel', 'timedelta2_rev_ip-app-device-os-channel', 
           'timemedian_app', 'nunique_app-os_app', 'totalCountByDay_app-device-os-channel', 'timedelta2_ip-app-device-os']

BEST_FEATURES_504 = ['app', 'timedelta_rev_ip-app-device-os', 'hour_min', 'device', 
                     'timeDD_rev_ip-app-device-os', 'nunique_ip-app_ip', 
                     'nunique_ip-app-device-os_ip-os', 'nunique_ip-os_ip', 
                     'timediff-minmax_app-channel', 'totalcount_app', 
                     'nunique_app-device-os-channel_app-channel', 'timeskew_ip', 
                     'nunique_device-os-channel_channel', 'totalcount_ip-app', 
                     'nunique_ip-app-device-channel_ip-channel', 'timevar_app-os-channel', 
                     'timedelta2_app-device-os-channel', 'timeDD_device-os', 
                     'timeDD_rev_ip-app-device-os-channel', 'nunique_ip-app-device_ip-device', 
                     'totalcount_ip-device', 'timedelta_ip', 'nunique_device-channel_channel', 
                     'totalCountByDay_ip-app-device', 'sameClickTimeCount_device-os', 
                     'timeskew_app-device', 'nunique_ip-app-device-channel_ip-app', 
                     'nunique_ip-device-os_os', 'totalCountByDay_app-os-channel', 
                     'timedelta2_rev_ip-app-device-os-channel', 'timemedian_app', 
                     'nunique_app-os_app', 'totalCountByDay_app-device-os-channel', 
                     'timedelta2_ip-app-device-os', 'timedelta_rev_ip-app-os', 
                     'timedelta_rev_ip-app-device-os-channel', 'timedelta_rev_ip-app-device', 
                     'timevar_ip', 'nunique_ip-device-os_ip-os', 'timedelta_rev_ip-os-channel', 
                     'totalRatioByDay_ip-device',]











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
    
    send_line(f'START {fname}  time: {elapsed_minute():.2f}min')
    
    return

def end(fname):
    
    print("""
#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(fname))
    print('time: {:.2f}min'.format( elapsed_minute() ))
    
    send_line(f'FINISH {fname}  time: {elapsed_minute():.2f}min')
    
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
    
    kf = KFold(n_splits=split_size)
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
        df.iloc[val_index].to_pickle(f'{path}/{i:03d}.p')
    return

def read_pickles(path, col=None):
    if col is None:
        df = pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*')))])
    else:
        df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
    return df

def to_feathers(df, path, split_size=3, inplace=True):
    """
    path = '../output/mydf'
    
    wirte '../output/mydf/0.f'
          '../output/mydf/1.f'
          '../output/mydf/2.f'
    
    """
    if inplace==True:
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.reset_index(drop=True)
    gc.collect()
    mkdir_p(path)
    
    kf = KFold(n_splits=split_size)
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
        df.iloc[val_index].to_feather(f'{path}/{i:03d}.f')
    return

def read_feathers(path, col=None):
    if col is None:
        df = pd.concat([pd.read_feather(f) for f in tqdm(sorted(glob(path+'/*')))])
    else:
        df = pd.concat([pd.read_feather(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
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

# =============================================================================
# other API
# =============================================================================
def submit(file_path, comment='from API'):
    os.system('kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f {} -m "{}"'.format(file_path, comment))

import requests
def send_line(message):
    
    line_notify_token = 'yII9fbfGF13HBMtV6EcHNfRhFGniqDfiqMbpZm89lTd'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}  # 発行したトークン
    requests.post(line_notify_api, data=payload, headers=headers)




