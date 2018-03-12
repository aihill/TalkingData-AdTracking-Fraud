#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 20:19:28 2018

@author: Kazuki
"""

import utils


train = utils.read_pickles('../data/train').sort_values(utils.sort_keys)


train[['is_attributed']].to_pickle('../data/label.p')

