#!/usr/bin/python3 -W ignore

##!/usr/local/bin/python3

import sys

import scipy
import scipy.io.wavfile
import numpy as np

from tensorflow.keras.models import load_model


def load_spectrogram(fn):

    data = load_wav(fn)
    return spectrogram(data)

def load_wav(fn):
    Fs, data = scipy.io.wavfile.read(fn)
    return data


#########################################################################

print('')
if len(sys.argv) < 3:
    print("usage: ptest3.py model.hdf5 test.wav")
    sys.exit()

model_fn = sys.argv[1]
model = load_model(model_fn)
print('loaded model', model_fn)
model.summary()
print('')

#x = load_wav('pos/F0YLMYYYYE.v1.wav')
x = load_wav(sys.argv[2])

if x.dtype == 'int16':
    x = np.divide(x, 2**15).astype('float32')

#x = x[0:6145]
x = np.reshape(x, (x.shape[0],1)).T

print('')

i=0
N = x.shape[1]
print(N)
stride = 512
vals = []
while i <= (N-4096):
    y = model.predict(x[..., i:i+4096], verbose=0)
    vals.append(y[0,0])
    print(i, y[0,0])

    i += stride

    #if y[0,0] > 0.9 and i < 20000:
    #    print(i, y[0,0])


#sys.exit()
import matplotlib.pyplot as plt

plt.figure(1)

xv = range(0, N)
plt.subplot(411)
plt.plot(xv,x[0])

plt.subplot(412)
vv = range(0, int((N-4096)/stride)+1)
vN = min(len(vv), len(vals))
plt.plot(vv[:vN],vals[:vN])

plt.show()

#plt.subplot(412)
#plt.plot(np.histogram(vals))

#binsT = np.asarray(notes[0]).T
##binsT[20] *= 1000
#plt.subplot(411)
#plt.imshow(binsT, cmap='hot', interpolation='nearest', aspect='auto')
#
#binsT = an.T
#plt.subplot(412)
#plt.imshow(binsT, cmap='hot', interpolation='nearest', aspect='auto')








# then, train on more samples with background mixed in, varying volume levels

# ok, now take this model and see how it does to pinpoint notes in pad-loud-soft.wav

# then, see how we do with same & different notes with partial/concurrent overlap

# add in percussive onset detection to improve detection/localization of notes.
# adjust optimizer to reduce false negatives.

# filtering has got to be in here somewhere ... train NN on spectrogram ...
# get on with estimating cpu load at run-time
