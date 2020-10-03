#!/usr/bin/python3
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from dsp import utils as u


# TODO read these from metadata/cfg file along with model somehow
SR = 44100  # sample rate
SPTGRM_WINDOW_SIZE = 512  # results in half this many frequency bins
SPTGRM_STRIDE = 128    # spectrogram stride,  ~ 3ms
SPTGRM_SAMPLEN = 32768  # spectrogram is created from this length of sample, 740ms in this case
map_vector_labels = None


if len(sys.argv) < 3:
    print("usage: ptest3.py model.hdf5 foo (loads foo.wav and foo.labels if present)")
    sys.exit()

model_fn = sys.argv[1]
wav_fn = sys.argv[2] + '.wav'
labels_fn = sys.argv[2] + '.labels'

x_audio = u.load_wav(wav_fn)
print('wav file loaded with %d samples' % (len(x_audio)))

if os.path.isfile(labels_fn):
    print('found labels in', labels_fn)
else:
    labels_fn = None
print('')


##########################################

import tensorflow as tf
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


model = load_model(model_fn)
print('loaded model', model_fn)
model.summary()
print('')
print('')


label2offsets = None
if labels_fn:
    label2offsets = {}
    for line in open(labels_fn):
        (offset, labels) = line.strip().split(':')
        offset = int(offset)
        for l in labels.split(','):
            offsets = label2offsets.get(l, [])
            offsets.append(offset)
            label2offsets[l] = offsets


# subsample for sake of plotting
subsamp = max(1, int(math.log(len(x_audio), 2)) - 17)
x_audio_plot = x_audio[0::subsamp]
xdpts = np.arange(0, len(x_audio_plot)) * subsamp # xdpts: x-axis data points == sample #


# compute spectrogram, evaluate with model and add to plot dataset one at a time, i.e. no batching
predictedlabel2offsets = {}
for i in range(0, len(x_audio)-SPTGRM_SAMPLEN, SPTGRM_STRIDE*2):
    if i%1000==0:
        print('%.2f%%' % (i/len(x_audio)*100))

    d = x_audio[i:i + SPTGRM_SAMPLEN]
    sptgm = u.spectrogram(d, SPTGRM_WINDOW_SIZE, SPTGRM_STRIDE)
    x = np.array(sptgm)[np.newaxis, ..., np.newaxis]
    print(x.shape)
    # y = model.predict(x, verbose=0)
    y = model(x)
    print(y)
    # map y to labels





#
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
