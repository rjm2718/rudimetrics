#!/usr/bin/python3

import random
import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, Conv1D, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import TensorBoard

from dsp import utils as u


# load all wav files from ./neg and ./pos
# each wav 4096 samples of raw audio
# return arrays of dims (m_train, 4096),(m_train,) and (m_test, 4096),(m_test,):
def load_data(pct_train=0.9, max_files=None):

    fl_neg = glob.glob('neg/*.wav')
    fl_pos = glob.glob('pos/*.wav')

    if max_files:
        F = len(fl_neg) + len(fl_pos)
        if F > max_files:
            pr = max_files / F # reduce file lists by this ratio
            random.shuffle(fl_neg)
            random.shuffle(fl_pos)
            fl_neg = fl_neg[0:(int(len(fl_neg)*pr))]
            fl_pos = fl_pos[0:(int(len(fl_pos)*pr))]

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    it_neg = int(pct_train * len(fl_neg))
    it_pos = int(pct_train * len(fl_pos))

    def s2x(X, fl, v):
        for l in map(u.load_spectrogram, fl):
            l = np.insert(l, 0, v, axis=0)
            X.append(l)

    s2x(X_train, fl_neg[0:it_neg], 0)
    s2x(X_train, fl_pos[0:it_pos], 1)
    X_train = np.asarray(X_train)
    np.random.shuffle(X_train)
    Y_train = X_train[...,0,0].astype('int32')
    X_train = X_train[...,1:,:].astype('float32')
    #print(X_train.shape)
    #print(Y_train.shape)

    s2x(X_test, fl_neg[it_neg:], 0)
    s2x(X_test, fl_pos[it_pos:], 1)
    X_test = np.asarray(X_test)
    np.random.shuffle(X_test)
    Y_test = X_test[...,0,0].astype('int32')
    X_test = X_test[...,1:,:].astype('float32')

    return X_train, Y_train, X_test, Y_test




def main():

    X_train, Y_train, X_test, Y_test = load_data(max_files=5000)

    print(X_train.shape)
    print(Y_train.shape)
    print('')
    #print(Y_train)

    u.validate_data(X_train)

    model = Sequential()

    model.add(BatchNormalization())

    model.add(Conv1D(filters=64,
                     kernel_size=7,
                     strides=2,
                     activation=tf.nn.relu,
                     #kernel_initializer='uniform',
                     #activity_regularizer=l2(.01),
                     #input_shape=(X_train.shape[1:])
                     ))

    model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Dense(64, activation=tf.nn.relu))
    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    model.add(Dense(16, activation=tf.nn.relu,
                        #kernel_initializer='uniform',
                        #kernel_regularizer=l2(0.01),
                        #activity_regularizer=l2(0.01)
                        ))
    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    model.add(Dense(1, activation=tf.nn.sigmoid))

    #optimizer = tf.keras.optimzizers.AdamOptimizer()
    optimizer = tf.keras.optimizers.RMSprop(lr=0.01)

    model.compile(optimizer=optimizer,
                  #loss='mse',
                  loss='binary_crossentropy',
                  metrics=['accuracy']) # XXX because of class imbalance, try different metric




    #tbCallBack = TensorBoard(log_dir='./tensorboard', histogram_freq=0, write_graph=True, write_images=True)

    # This builds the model for the first time:
    #model.fit(X_train, Y_train, epochs=2, steps_per_epoch=10, callbacks=[tbCallBack])
    #model.fit(X_train, Y_train, epochs=10, steps_per_epoch=20)
    model.fit(X_train, Y_train, epochs=10)

    #model.summary()
    #all_weights = []
    #for layer in model.layers:
    #   w = layer.get_weights()
    #   all_weights.append(w)
    #all_weights = np.array(all_weights)
    #print(all_weights)


    test_loss, test_acc = model.evaluate(X_test, Y_test)

    print('Test accuracy:', test_acc)

    tf.keras.models.save_model(model, 'train-model-simple-dense-spectrogram.hdf5')



if __name__ == '__main__':
    main()