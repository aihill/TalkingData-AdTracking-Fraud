#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:04:09 2018

@author: kazuki.onodera
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys
sys.path.append('/home/kazuki_onodera/Python')
import xgbextension as ex
#import xgboost as xgb
#from multiprocessing import Process, Pipe
import threading
from queue import Queue
import gc
#from time import sleep
import utils
utils.start(__file__)

SEED = np.random.randint(9999) #int(sys.argv[1])
FRAC = 0.1


np.random.seed(SEED)
print('seed :', SEED)

# =============================================================================
# def
# =============================================================================
df_queue = Queue()
#df_list = []
def sender(load_file):
    print('loading {} ...'.format(load_file))
    df_queue.put( pd.read_pickle(load_file).sample(frac=FRAC, random_state=SEED) )
#    df_list.append( pd.read_pickle(load_file).sample(frac=FRAC, random_state=SEED) )
    print('loaded {}'.format(load_file))

# =============================================================================
# load train
# =============================================================================

X = pd.concat([utils.read_pickles('../data/train').sample(frac=FRAC, random_state=SEED),
                   pd.read_pickle('../data/101_train.p').sample(frac=FRAC, random_state=SEED),
                   pd.read_pickle('../data/102_train.p').sample(frac=FRAC, random_state=SEED),
                   pd.read_pickle('../data/104_train.p').sample(frac=FRAC, random_state=SEED),
                   ], 
                  axis=1)
gc.collect()

# threading
threads = [None] * len(utils.comb)
for i, keys in enumerate(utils.comb):
    load_file = '../data/{}_train.p'.format('-'.join(keys))
    threads[i] = threading.Thread(target=sender, args=(load_file, ))
    threads[i].start()

#df_queue.join()

for t in threads:
    t.join()


X = pd.concat([X] + [df_queue.get() for i in utils.comb], axis=1)
#X = pd.concat([X] + df_list, axis=1)


# straight
#X = pd.concat([X]+[pd.read_pickle('../data/{}_train.p'.format('-'.join(keys))).sample(frac=FRAC, random_state=SEED) for keys in tqdm(utils.comb)],
#              axis=1)



gc.collect()

y = X.is_attributed
X.drop(['ip', 'app', 'device', 'os', 'channel', 'is_attributed', 'click_time', 'attributed_time'], 
           axis=1, inplace=True)
X.fillna(-1, inplace=True)

gc.collect()

print(X.columns.tolist())

# =============================================================================
# xgboost
# =============================================================================


param = {'colsample_bylebel': 0.8,
         'subsample': 0.6,
         'eta': 0.1,
         'eval_metric': 'auc',
         'max_depth': 4,
         'objective': 'binary:logistic',
         'silent': 1,
         'tree_method': 'hist',
         'nthread': 64,
         'seed': SEED}


gc.collect()


yhat, imp, ret = ex.stacking(X, y, param, 9999, nfold=5, esr=30)

imp.to_csv('imp_{}.csv'.format(datetime.today().date()), index=False)

# =============================================================================
# cv
# =============================================================================

#dtrain = xgb.DMatrix(train, y)
#
#cv = xgb.cv(param, dtrain, 9999, 
#            nfold=5, early_stopping_rounds=50, verbose_eval=5)

#==============================================================================
utils.end(__file__)

