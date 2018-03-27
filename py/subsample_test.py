#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 23:07:52 2018

@author: Kazuki
"""

import numpy as np
from collections import defaultdict

ratio = 0.15
N = 1000
LOOP = 20

li = list(range(N))
di = defaultdict(int)

for i in range(LOOP):
    choice = np.random.choice(li, replace=False, size=int(N*ratio))
    for j in choice:
        di[j] +=1

print(len(di)/N)


