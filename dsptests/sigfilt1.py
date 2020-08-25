#!/usr/bin/python3

# test bandpass filter 200hz to 2Khz .....


# t = numpy.arange(5000)
# w = t * 2 * numpy.pi
# v = numpy.sin(w / 100) + numpy.sin(w / 120)
# import matplotlib.pyplot as plt
# plt.plot(v)
# plt.show()



import numpy
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt
import random
from numpy.core.multiarray import arange
from scipy.fftpack import fft

from scipy import signal
from scipy.signal import kaiserord, lfilter, firwin, freqz



def firConv(s, coefs):

    o = []

    for i in range(0, len(s)):

        av = 0.0
        for k in range(0, len(coefs)):

            if i-k >= 0: sv = s[i-k]
            else: sv = 0

            av += coefs[k] * sv

        #s[i] = int(av)
        #s[i] = av
        o.append(av)

    for i in range(0, len(s)):
        s[i] = o[i]




def plotSpectr(s, Fs, color='g'):

    N = len(s)
    T = N / float(Fs)
    freqs = arange(N/16) / T
    Y = fft(s) / N
    Y = Y[range(N/16)]

    plt.plot(freqs, abs(Y), color)



#####################################################################3

N = 4096

fn = 'sticks.wav'
Fs, s = scipy.io.wavfile.read(fn)


#Fs = 44100
nyq_rate = Fs / 2.

#width=100/nyq_rate
#ripple_db = 60.0
#n_taps, beta = kaiserord(ripple_db, width)
taps = firwin(150, [0.04, 0.05], pass_zero=False, window='hamming') #, window=('kaiser', beta))

#taps = numpy.array([1] * 16)

#taps = [0.000126, 0.000126] 

#taps = signal.hann(161)


taps = taps/float(sum(taps))

print len(taps)
print taps



#   n = numpy.arange(N)

#   fi = 555.0
#   w = n * 2 * numpy.pi * fi / Fs
#   s = numpy.sin(w)

#   fi = Fs/200.0 #333.3
#   w = n * 2 * numpy.pi * fi / Fs
#   s += numpy.sin(w)

#   fi = 1002.0
#   w = n * 2 * numpy.pi * fi / Fs
#   s += numpy.sin(w)

#   s *= 10000


plt.figure(num=1, figsize=(20,10))

plt.subplot(221)
plt.plot(arange(0, N), s, 'b')

plt.subplot(222)
plotSpectr(s, Fs, 'b')



#firConv(s, taps)
#s = lfilter(taps, 1.0, s)
s = signal.convolve(s, taps, mode='same')

plt.subplot(223)
plt.plot(arange(0, N), s, 'r')
plt.subplot(224)
plotSpectr(s, Fs, 'r')

plt.show()











