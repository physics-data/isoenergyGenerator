#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import h5py
import sys
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import numpy as np

TWO_FOLD_DATA=[]
TWO_FOLD_PROBLEM=[]
FOUR_FOLD_DATA=[]
FOUR_FOLD_PROBLEM=[]

def read_data(number, prefix, dataset):
    with h5py.File(f'{prefix}/{number:04}.h5', 'r') as h:
        return h[dataset][()]

with Pool(250) as p:
    TWO_FOLD_DATA = list(p.imap(partial(read_data, prefix='dos-momentum-2fold/h5', dataset='/isoE'), range(10000)))
    print(1)
    TWO_FOLD_PROBLEM = list(p.imap(partial(read_data, prefix='dos-momentum-2fold/scatter', dataset='/QPI'), range(10000)))
    print(2)
    FOUR_FOLD_DATA = list(p.imap(partial(read_data, prefix='dos-momentum-4fold/h5', dataset='/isoE'), range(10000)))
    print(3)
    FOUR_FOLD_PROBLEM = list(p.imap(partial(read_data, prefix='dos-momentum-4fold/scatter', dataset='/QPI'), range(10000)))
    print(4)

# problem: first 1000 + 1000 QPI data
with h5py.File(f'problem2.h5', 'w') as h:
    for i in tqdm(range(1000)):
        h.create_dataset(str(2 * i).zfill(4)+'/QPI',data=TWO_FOLD_PROBLEM[i],dtype='float32', compression='gzip')
        h.create_dataset(str(2 * i + 1).zfill(4)+'/QPI',data=FOUR_FOLD_PROBLEM[i],dtype='float32', compression='gzip')

# answer: first 1000 + 1000 isoE data
with h5py.File(f'anwser2.h5', 'w') as h:
    for i in tqdm(range(1000)):
        h.create_dataset(str(2 * i).zfill(4)+'/isoE',data=TWO_FOLD_DATA[i],dtype='float32', compression='gzip')
        h.create_dataset(str(2 * i + 1).zfill(4)+'/isoE',data=FOUR_FOLD_DATA[i],dtype='float32', compression='gzip')

# exmple: some random 2000 isoE data
with h5py.File(f'example2.h5', 'w') as h:
    for i in tqdm(range(2000)):
        h.create_dataset(str(i).zfill(4)+'/isoE',data=np.zeros((1005, 1005)),dtype='float32', compression='gzip')

# train: remaining 9000 + 9000 QPI + isoE data
with h5py.File(f'train2.h5', 'w') as h:
    for i in tqdm(range(1000, 10000)):
        h.create_dataset(str(2 * i).zfill(4)+'/isoE',data=TWO_FOLD_DATA[i],dtype='float32', compression='gzip')
        h.create_dataset(str(2 * i).zfill(4)+'/QPI',data=TWO_FOLD_PROBLEM[i],dtype='float32', compression='gzip')
        h.create_dataset(str(2 * i + 1).zfill(4)+'/isoE',data=FOUR_FOLD_DATA[i],dtype='float32', compression='gzip')
        h.create_dataset(str(2 * i + 1).zfill(4)+'/QPI',data=FOUR_FOLD_PROBLEM[i],dtype='float32', compression='gzip')

