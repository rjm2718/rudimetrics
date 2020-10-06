#!/usr/bin/python3
import math
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import json

from dsp import utils as u

def main():

    if len(sys.argv) < 3:
        print("usage: ptest3.py modeldir foo (loads foo.wav and foo.labels if present)")
        sys.exit()

    model_fn = sys.argv[1]
    if sys.argv[2].endswith('.wav'):
        sys.argv[2] = sys.argv[2][0:-4]
    wav_fn = sys.argv[2] + '.wav'
    labels_fn = sys.argv[2] + '.labels'

    cfg = json.load(open(model_fn + os.sep + 'r.cfg.json'))
    print('loaded config from model:', cfg)
    SR = cfg['SR']
    SPTGRM_WINDOW_SIZE = cfg['SPTGRM_WINDOW_SIZE']
    SPTGRM_STRIDE = cfg['SPTGRM_STRIDE']
    SPTGRM_SAMPLEN = cfg['SPTGRM_SAMPLEN']
    label_vector_map = cfg['label_vector_map']


    if os.path.isfile(labels_fn):
        print('found labels in', labels_fn)
    else:
        labels_fn = None


    x_audio = u.load_wav(wav_fn)
    # subsample for sake of plotting
    subsamp = max(1, int(math.log(len(x_audio), 2)) - 17)
    x_audio_plot = x_audio[0::subsamp]
    xa_dpts = np.arange(0, len(x_audio_plot)) * subsamp # xdpts: x-axis data points == sample #
    print('wav file loaded with %d samples' % (len(x_audio)))

    print('computing spectrogram')
    x_sptgm = u.spectrogram(x_audio, SPTGRM_WINDOW_SIZE, SPTGRM_STRIDE)
    # xs_dpts = np.arange(0, len(x_sptgm)) * SPTGRM_STRIDE
    print('spectrogram shape', x_sptgm.shape)

    expected_sptg_len = (len(x_audio)-SPTGRM_WINDOW_SIZE)/SPTGRM_STRIDE + 1
    print('expected spectrogram len =', expected_sptg_len)
    # return

    # # given an offset, spectrogram slice is offset/SPTGRM_STRIDE to +(SPTGRM_SAMPLEN-SPTGRM_WINDOW_SIZE)/SPTGRM_STRIDE+1
    # spectrogram_len = int((SPTGRM_SAMPLEN - SPTGRM_WINDOW_SIZE) / SPTGRM_STRIDE + 1)
    # si0 = int(offset / SPTGRM_STRIDE)
    # si1 = si0 + spectrogram_len

    # fig, axes = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    # axes[0].plot(x_audio)
    # axes[0].margins(0.)
    #
    # x_sptgm = np.power(x_sptgm, .5)
    # zoom = int(len(x_audio)/len(x_sptgm))
    # x_sptgm = scipy.ndimage.zoom(x_sptgm, (zoom,1), order=0)
    # axes[1].imshow(x_sptgm.T, cmap='hot', interpolation='nearest', aspect='auto', origin='lower')
    # axes[1].margins(0.)
    #



    ##########################################

    import tensorflow as tf
    from tensorflow.keras.models import load_model

    # test without GPU:  $ CUDA_VISIBLE_DEVICES=-1 ./train-model-simple-dense-spectrogram.py ...
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

    label2offsets = {}
    if labels_fn:
        for line in open(labels_fn):
            (offset, labels) = line.strip().split(':')
            offset = int(offset)
            for l in labels.split(','):
                offsets = label2offsets.get(l, [])
                offsets.append(offset)
                label2offsets[l] = offsets



    sptgm_shape = model.layers[0].input.shape[1:3]
    print(sptgm_shape)

    # evaluate with model and add to plot dataset one at a time, i.e. no batching
    predictedlabel2offsets = {l:[] for l in label_vector_map.keys()} # map label to list of (offset,predicted-value) pairs, split later for graph

    PRDCT_BATCHLEN = 32 # in practice we just don't want too much latency ... shoot for .5 sec at worst
    xs_dpts = []
    t1 = time.time()
    for i in range(0, len(x_sptgm) - sptgm_shape[0], PRDCT_BATCHLEN):
        if i%(PRDCT_BATCHLEN*10)==0:
            print('%.1f%%'%(i/(len(x_sptgm)-sptgm_shape[0])*100))

        batch = []
        for j in range(0, PRDCT_BATCHLEN):
            x = np.array(x_sptgm[i:i+sptgm_shape[0]])[..., np.newaxis]
            batch.append(x)
            xs_dpts.append((i+j)*SPTGRM_STRIDE)

        x = np.array(batch)
        y = model.predict(x, batch_size=PRDCT_BATCHLEN, verbose=0)

        for l,y_i in label_vector_map.items():
            predictedlabel2offsets[l].extend(y[...,y_i])

    for l, y_i in label_vector_map.items():
        print(l, len(predictedlabel2offsets[l]))

    t2 = time.time()
    prate = (len(x_sptgm) - sptgm_shape[0])/(t2-t1)
    print('%.1f predictions/sec' % (prate))

    fig, axes = plt.subplots(len(label_vector_map)+1, 1, constrained_layout=True, sharex=True)
    axes[0].plot(xa_dpts, x_audio_plot, linewidth=1)
    axes[0].margins(0.)

    for l in label2offsets:
        offsets = label2offsets[l]
        axes[0].scatter(offsets, np.ones(len(offsets)), marker='v', label=l)
        axes[0].legend()

    print(len(xs_dpts))
    ai = 1
    for l in predictedlabel2offsets:
        axes[ai].plot(xs_dpts, predictedlabel2offsets[l], label=l)
        axes[ai].legend()
        axes[ai].margins(0.)
        ai += 1


    plt.axis('tight')
    plt.margins(0.05, 0)
    plt.show()

if __name__ == '__main__':
    main()
