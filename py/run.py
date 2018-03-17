#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 19:37:49 2018

@author: Kazuki
"""

import os
import sys
argv = sys.argv

file = argv[1]

os.system('nohup python -u {} > log.txt &'.format(file))

