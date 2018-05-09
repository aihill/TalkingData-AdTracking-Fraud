#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 07:44:05 2018

@author: Kazuki
"""


import pandas as pd
import utils
#utils.start(__file__)

SUBMIT_FILE_PATH1 = '../output/504-2.csv.gz'
SUBMIT_FILE_PATH2 = '../output/submission_final4.csv'

SUBMIT_FILE_PATH = '../output/504-2_final4.csv.gz'
COMMENT = f"504-2(0.9805) + 0.9812"

EXE_SUBMIT = True


sub1 = pd.read_csv(SUBMIT_FILE_PATH1).set_index('click_id')
sub2 = pd.read_csv(SUBMIT_FILE_PATH2).set_index('click_id')

sub = sub1.rank() + sub2.rank()

sub['is_attributed'] /= sub['is_attributed'].max()

sub.reset_index(inplace=True)
sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

# =============================================================================
# submission
# =============================================================================
if EXE_SUBMIT:
    print('submit')
    utils.submit(SUBMIT_FILE_PATH, COMMENT)


#==============================================================================
utils.end(__file__)


