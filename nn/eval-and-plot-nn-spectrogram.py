#!/usr/bin/python3
import sys

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from dsp import utils as u


# TODO read these from metadata/cfg file along with model somehow
SR = 44100  # sample rate
SPTGRM_WINDOW_SIZE = 512  # results in half this many frequency bins
SPTGRM_STRIDE = 128    # spectrogram stride,  ~ 3ms
SPTGRM_SAMPLEN = 32768  # spectrogram is created from this length of sample, 740ms in this case



print('')
if len(sys.argv) < 3:
    print("usage: ptest3.py model.hdf5 test.wav")
    sys.exit()

model_fn = sys.argv[1]
wav_fn = sys.argv[2]

model = load_model(model_fn)
print('loaded model', model_fn)
model.summary()
print('')
print('')

x_audio = u.load_wav(wav_fn)
x_spctgm = []

print(len(x_audio))
for i in range(0, len(x_audio)-SPTGRM_SAMPLEN, int(SPTGRM_STRIDE/2)):
    if i%1000==0:
        print('%.2f%%' % (i/len(x_audio)*100))
    d = x_audio[i:i + SPTGRM_SAMPLEN]
    x_spctgm.append( u.spectrogram(d, SPTGRM_WINDOW_SIZE, SPTGRM_STRIDE) )

x_spctgm = np.array(x_spctgm, copy=False)[..., np.newaxis]
print('x_spctgm shape', x_spctgm.shape)
print('')


# # plot wav data on upper plot, prediction values below.
# y = np.zeros((len(x_audio), 2)) # two categories for now
#
# while i <= (N-model_dims[0]):
#     xs = x_[:,i:i+model_dims[0],:]
#     #print(np.max(np.abs(xs)))
#     y = model.predict(xs, verbose=0)
#     y = np.power(y, .1)
#     vals[i] = y[0,]*10
#     #vals.append(y[0,0]*100)
#     #print(i, int(y[0,0]*100))
#
#     i += stride
#
#     #if y[0,0] > 0.9 and i < 20000:
#     #    print(i, y[0,0])
#
#
# fig, axes = plt.subplots(3,1, constrained_layout=True)
# xp = np.multiply(x, 10)
# xp = np.power(xp, .3).T
# axes[0].imshow(xp, cmap='hot', interpolation='nearest', aspect='auto')
# axes[0].margins(0.)
#
# axes[1].plot(x_audio.T)
# axes[1].margins(0.)
#
# axes[2].plot(vals)
# axes[2].margins(0.)
#
# plt.axis('tight')
# plt.margins(0.05, 0)
# plt.show()
