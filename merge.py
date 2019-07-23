#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import h5py
import sys

mode = sys.argv[1]
inDir = sys.argv[2]
outDir = sys.argv[3]

if mode == '0':
    # problem
    with h5py.File(outDir+'problem.h5','w') as opt:
        for i in range(900):
            if i%100==0:
                print(i)
            ipt = h5py.File(inDir+str(i+100).zfill(4)+'.h5')
            opt.create_dataset(str(i).zfill(4)+'/QPI',data=ipt['QPI'],dtype='float32', compression='gzip')
            ipt.close()
    print('problem data generated')

elif mode == '1':
    #example
    with h5py.File(outputDir+'example.h5','w') as opt:
        for i in range(900):
            if i%100==0:
                print(i)
            ipt = h5py.File(dataIn/'+str(i+1000)+'.h5')
            opt.create_dataset(str(i).zfill4)+'isoE',data=ipt['isoE'],dtype=float32,compression='gzip')
            ipt.close()
    print('example data generated')
else:
    #train
    with h5py.File(outDir+'train.h5','w') as opt:
        for i in range(9000):
            if i%1000==0:
                print(i)
            ipt = h5py.File(inDir+str(i+1000)+'.h5')
            ipt2 = h5py.File('dataIn/'+str(i+1000)+'.h5')
            opt.create_dataset(str(i).zfill(4)+'/QPI',data=ipt['QPI'],dtype = 'float32', compression='gzip')
            opt.create_dataset(str(i).zfill(4)+'/isoE',data=ipt2['isoE'],dtype = 'float32', compression='gzip')
            ipt.close()
            ipt2.close()
    print('train data generated')

