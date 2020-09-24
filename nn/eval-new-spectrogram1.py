#!/usr/bin/python3
import sys

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from dsp import utils as u



print('')
if len(sys.argv) < 3:
    print("usage: ptest3.py model.hdf5 test.wav")
    sys.exit()

model_fn = sys.argv[1]
model = load_model(model_fn)
print('loaded model', model_fn)
#model.summary()
print('')

x_audio = u.load_wav(sys.argv[2])

x = u.load_spectrogram(sys.argv[2])
print(x.shape)
print('')

#for i in range(x.shape[0]):
#    print(int(np.max(x[i])*100))
#sys.exit()

model_dims = (121, 129)

i=0
N = x.shape[0]
stride = 15
vals = np.zeros(N)
x_ = np.expand_dims(x, axis=0)
print(x_.shape)
while i <= (N-model_dims[0]):
    xs = x_[:,i:i+model_dims[0],:]
    #print(np.max(np.abs(xs)))
    y = model.predict(xs, verbose=0)
    y = np.power(y, .1)
    vals[i] = y[0,]*10
    #vals.append(y[0,0]*100)
    #print(i, int(y[0,0]*100))

    i += stride

    #if y[0,0] > 0.9 and i < 20000:
    #    print(i, y[0,0])


fig, axes = plt.subplots(3,1, constrained_layout=True)
xp = np.multiply(x, 10)
xp = np.power(xp, .3).T
axes[0].imshow(xp, cmap='hot', interpolation='nearest', aspect='auto')
axes[0].margins(0.)

axes[1].plot(x_audio.T)
axes[1].margins(0.)

axes[2].plot(vals)
axes[2].margins(0.)

plt.axis('tight')
plt.margins(0.05, 0)
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

