#!/usr/local/bin/python3

# take list of wav files on cmd line, normalize bit depth, length, channels, volume.
# new version of wav is written to current directory with random file name.
#
# e.g. ./mktraining.py neg_originals/*.wav ; mv *.wav neg/
# 

import wavio
import scipy
import scipy.io.wavfile
import numpy as np

import string
import random
import glob, sys

import utils as u

# for each file:
#  load with scipy.io.wavfile.read, or wavio.read, whichever works, return Fs,data.
#  If Fs != 44.1k, abort
#  If data is 2 channels, average down to single channel
#  normalize volume

# output variations of file:
#  - couple random volume reductions


def resize(data, L=4096):

    if data.shape[0] < L:
        pl = L - data.shape[0]
        data = np.pad(data, (0, pl), mode='constant')

    else:
        data = data[:L]

    return data

def loadWav(fn):

    data = u.load_wav(fn)
    data = u.normalize(data)
    data = resize(data)
    return data




for fn in sys.argv[1:]:

    data = loadWav(fn)
    print(fn)

    fn_new = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    scipy.io.wavfile.write(fn_new + '.0.wav', 44100, data)

    data2 = np.multiply(data, 0.2)
    scipy.io.wavfile.write(fn_new + '.v1.wav', 44100, data2)
    scipy.io.wavfile.write(fn_new + '.v2.wav', 44100, np.multiply(data2, random.random()/2))
    scipy.io.wavfile.write(fn_new + '.v3.wav', 44100, np.multiply(data, random.random()))

    scipy.io.wavfile.write(fn_new + '.n1.wav', 44100, np.asarray(list(map(lambda x: x + (random.random() - 0.5)/80., data2))).astype('float32')) 
    scipy.io.wavfile.write(fn_new + '.n2.wav', 44100, np.asarray(list(map(lambda x: x + (random.random() - 0.5)/20., data))).astype('float32'))


# then, train on more samples with background mixed in, varying volume levels

# ok, now take this model and see how it does to pinpoint notes in pad-loud-soft.wav

# then, see how we do with same & different notes with partial/concurrent overlap

# add in percussive onset detection to improve detection/localization of notes.
# adjust optimizer to reduce false negatives.

# filtering has got to be in here somewhere ... train NN on spectrogram ...
# get on with estimating cpu load at run-time

