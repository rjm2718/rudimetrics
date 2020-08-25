#!/usr/bin/python3

import wavio
import scipy
import scipy.io.wavfile
import numpy as np

import string
import random
import glob, sys


Fs, data = scipy.io.wavfile.read('misc-noise-floor1.wav')
#Fs, data = scipy.io.wavfile.read('backgrounds/background1.wav')

if data.dtype == 'int16':
    data = np.divide(data, 2**15).astype('float32')


N = data.shape[0]
i0 = 0
stride=512
while i0 <= (N-4096):
    scipy.io.wavfile.write('sxnf.{}.wav'.format(i0), 44100, data[i0:i0+4096])
    scipy.io.wavfile.write('sxnf.{}n.wav'.format(i0), 44100, np.multiply(data[i0:i0+4096], 1+10*random.random()).astype('float32'))
    i0 += stride


