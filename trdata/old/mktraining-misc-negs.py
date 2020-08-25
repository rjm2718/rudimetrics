#!/usr/local/bin/python3

# take list of wav files on cmd line, normalize bit depth, length, channels, volume.
# new version of wav is written to current directory with random file name.
#
# e.g. ./mktraining.py neg_originals/*.wav ; mv *.wav neg/
# 

# background samples, pos samples, neg samples
#
# neg:
#  + all neg samples
#  + background snippets
#  + neg samples plus background snippets
#  + random volume changes of above
#  + stretched/compressed in time of above
#  + generated white noise
#  + generated tones
#  + synthesize impulse/step functions of varying onsets
#
# pos:
#  + all pos samples
#  + pos samples with plus background snippets
#  + pos samples plus tones, white noise
#  + random volume changes of above
#
# pad all training examples to same length
#
# how much should we strive to detect overlapping notes?
#  - focus on a simple detection network experiment to start
#


import wavio
import scipy
import scipy.io.wavfile
import numpy as np

import string
import random
import glob, sys


# generate more negative samples:
#  random splices from background tracks
#  generated spike functions

fn_new = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

for bgfn in glob.glob('backgrounds/*.wav'):

    Fs, data = scipy.io.wavfile.read(bgfn)
    for _ in range(10):
        i0 = random.randint(0, len(data) - 4096)
        scipy.io.wavfile.write(fn_new + '.b1.wav', 44100, data[i0:i0+4096])
    

data = [0.] * 4096
scipy.io.wavfile.write(fn_new + '.s1.wav', 44100, np.asarray(data).astype('float32'))
data = [0.01, -0.01] * 2048
scipy.io.wavfile.write(fn_new + '.nx.wav', 44100, np.asarray(list(map(lambda x: x + 1.9*(random.random() - 0.5), data))).astype('float32'))
scipy.io.wavfile.write(fn_new + '.nx2.wav', 44100, np.asarray(list(map(lambda x: x + .5*(random.random() - 0.5), data))).astype('float32'))

def noisyspike(z1=10):
    sw = random.randint(10, 40)
    d = [0.] * z1
    d.extend( [.5] * 2 )
    d.extend( [-.5] * 2 )
    d.extend( [.99] * sw )
    d.extend( [-.99] * sw )
    d.extend( [.5] * int(sw/4) )
    d.extend( [-.5] * int(sw/4) )
    d.extend( [.25] * 5 )
    d.extend( [-.25] * 5 )
    d.extend( [.1] * 10 )
    d.extend( [-.1] * 10 )
    d.extend( [0] * (4096-len(d)) )
    d = list(map(lambda x: x + (random.random() - 0.5)/random.randint(5,25), d))
    return np.asarray(d).astype('float32')

scipy.io.wavfile.write(fn_new + '.p1.wav', 44100, noisyspike(10))
scipy.io.wavfile.write(fn_new + '.p2.wav', 44100, noisyspike(50))
scipy.io.wavfile.write(fn_new + '.p3.wav', 44100, noisyspike(450))
scipy.io.wavfile.write(fn_new + '.p4.wav', 44100, noisyspike(1450))
scipy.io.wavfile.write(fn_new + '.p5.wav', 44100, noisyspike(2500))



# then, train on more samples with background mixed in, varying volume levels

# ok, now take this model and see how it does to pinpoint notes in pad-loud-soft.wav

# then, see how we do with same & different notes with partial/concurrent overlap

# add in percussive onset detection to improve detection/localization of notes.
# adjust optimizer to reduce false negatives.

# filtering has got to be in here somewhere ... train NN on spectrogram ...
# get on with estimating cpu load at run-time

