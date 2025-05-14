#!/usr/bin/python3
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

model_fn = sys.argv[1]

model = load_model(model_fn)
print('loaded model', model_fn)
model.summary()
print('')
print('')


xs = np.zeros((1, 253,257,1))
y = model.predict(xs)

print(y)
print(y.shape)
