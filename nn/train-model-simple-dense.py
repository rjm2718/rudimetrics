#!/usr/bin/python3 -W ignore

# #!/usr/local/bin/python3

import tensorflow as tf
from tensorflow import keras

import scipy
import scipy.io.wavfile
import numpy as np
#   import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, Conv1D

import sys
import random
import glob

import utils as u


# load all wav files from ./neg and ./pos
# each wav 4096 samples of raw audio
# return arrays of dims (m_train, 4096),(m_train,) and (m_test, 4096),(m_test,):
def load_data(pct_train=0.9):

    fl_neg = glob.glob('neg/*.wav')
    fl_pos = glob.glob('pos/*.wav')

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    it_neg = int(pct_train * len(fl_neg))
    it_pos = int(pct_train * len(fl_pos))

    def s2x(X, fl, v):
        for l in map(u.load_wav, fl):
            l = np.insert(l, 0, v)
            X.append(l)

    s2x(X_train, fl_neg[0:it_neg], 0)
    s2x(X_train, fl_pos[0:it_pos], 1)
    X_train = np.asarray(X_train)
    np.random.shuffle(X_train)
    Y_train = X_train[...,0].astype('int32')
    X_train = X_train[...,1:].astype('float32')

    s2x(X_test, fl_neg[it_neg:], 0)
    s2x(X_test, fl_pos[it_pos:], 1)
    X_test = np.asarray(X_test)
    np.random.shuffle(X_test)
    Y_test = X_test[...,0].astype('int32')
    X_test = X_test[...,1:].astype('float32')

    return X_train, Y_train, X_test, Y_test



#########################################################################

X_train, Y_train, X_test, Y_test = load_data()

u.validate_data(X_train)


model = Sequential()
#model.add(Conv1D(filters=196, kernel_size=15, strides=4))
model.add(Dense(64, activation=tf.nn.relu, input_shape=(X_train.shape[1],)))
model.add(Dropout(0.1))
model.add(Dense(16, activation=tf.nn.relu))
model.add(Dense(1, activation=tf.nn.sigmoid))

#print(model.get_weights())

#optimizer = tf.train.AdamOptimizer()
optimizer = tf.keras.optimizers.RMSprop(lr=0.001)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

#model.summary()

# This builds the model for the first time:
model.fit(X_train, Y_train, epochs=10)

test_loss, test_acc = model.evaluate(X_test, Y_test)

print('Test accuracy:', test_acc) # 0.98 ... that was easy 

tf.keras.models.save_model(model, 'train-model-simple-dense-raw-audio.hdf5')


# then, train on more samples with background mixed in, varying volume levels

# ok, now take this model and see how it does to pinpoint notes in pad-loud-soft.wav

# then, see how we do with same & different notes with partial/concurrent overlap

# add in percussive onset detection to improve detection/localization of notes.
# adjust optimizer to reduce false negatives.

# filtering has got to be in here somewhere ... train NN on spectrogram ...
# get on with estimating cpu load at run-time

