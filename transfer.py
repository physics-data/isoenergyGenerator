#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import h5py
import numpy as np
import sys

inDat = sys.argv[1]
outH5 = sys.argv[2]
N = 201
isoE = np.fromfile(inDat,dtype = 'float32', sep = '')
isoE = isoE.reshape([N,N])
with h5py.File(outH5,'w') as opt:
    opt.create_dataset('/isoE',data = isoE, compression = 'gzip')
