#!/usr/bin/env python3
# -*- coding: utf8 -*-

import h5py
import numpy as np
import sys

flag = int(sys.argv[1])
inputMom = sys.argv[2]
outputPos = sys.argv[3]
Nfft = 603
if flag == 0:
    with h5py.File(inputMom,'r') as ipt, h5py.File(outputPos,'w') as opt:
        isoE = np.array(ipt['/isoE'])
        k_fft = np.fft.fft2(isoE,[Nfft,Nfft])
        k_int = np.sum(isoE)
        k_fft_shift = np.fft.fftshift(k_fft)
        Dr = k_int**3 + 3 * k_int * np.abs(k_fft_shift)**2
        # opt['/QPI'] = Dr[(Nfft//2-100):(Nfft//2+101),(Nfft//2-100):(Nfft//2+101)]
        opt.create_dataset('/QPI',data = Dr, compression = 'gzip')
        #print(Dr.shape,isoE.shape,Dr[(Nfft//2-100):(Nfft//2+101),(Nfft//2-100):(Nfft//2+101)])
else:
    with h5py.File(inputMom,'r') as ipt, h5py.File(outputPos,'w') as opt:
        isoE = np.array(ipt['/isoE'])
        k_fft = np.fft.fft2(isoE,[Nfft,Nfft])
        k_int = np.sum(isoE)
        k_fft_shift = np.fft.fftshift(k_fft)
        Dr = k_int**3 - k_int * np.abs(k_fft_shift)**2
        # opt['/QPI'] = Dr[(Nfft//2-100):(Nfft//2+100),(Nfft//2-100):(Nfft//2+101)]
        opt.create_dataset('/QPI',data = Dr, compression = 'gzip')
