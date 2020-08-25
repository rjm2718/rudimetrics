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

fn = 'samples/roll.sticks1.20160626.wav'
Fs, s = scipy.io.wavfile.read(fn)
s = s[:30000]


#Fs = 44100
nyq_rate = Fs / 2.

#width=5/nyq_rate
#ripple_db = 60.0
#n_taps, beta = kaiserord(ripple_db, width)
#taps = firwin(512, [440, 448, 1940, 1970], pass_zero=False, window=('kaiser', beta), nyq=nyq_rate)

#taps = firwin(1024, [1945, 1965], pass_zero=False, width=10, nyq=nyq_rate)
taps = firwin(200, [440,450, 1950, 1960, 2930,2940], pass_zero=False, width=10, nyq=nyq_rate, window='hamming')
taps = firwin(101, [1950, 1960, 2930,2940], pass_zero=False, width=20, nyq=nyq_rate, window='hamming')

#taps = numpy.array([1] * 16)

#taps = [0.000126, 0.000126] 

#taps = signal.hann(161)

#taps = taps/float(sum(taps))
taps = taps * 5

print len(s)
print len(taps)
print taps


if True:
    plt.figure(1)
    plt.plot(taps, 'bo-', linewidth=2)
    plt.title('Filter Coefficients (%d taps)' % len(taps))
    plt.grid(True)

    plt.figure(2)
    plt.clf()
    w, h = freqz(taps, worN=8000)
    plt.plot((w/numpy.pi)*nyq_rate, numpy.absolute(h), linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency Response')
    #plt.ylim(-0.05, 200.5)
    plt.grid(True)

    plt.show()



plt.figure(1)
plt.plot(arange(0, len(s)), s, 'b')
plt.figure(2)
plt.clf()
w, h = freqz(s, worN=4096)
plt.plot((w/numpy.pi)*nyq_rate, numpy.absolute(h), linewidth=2)
plt.grid(True)


firConv(s, taps);
s2 = s
#s2 = signal.convolve(s, taps, mode='same')

plt.figure(3)
plt.plot(arange(0, len(s2)), s2, 'b')
plt.figure(4)
plt.clf()
w, h = freqz(s2, worN=4096)
plt.plot((w/numpy.pi)*nyq_rate, numpy.absolute(h), linewidth=2)
plt.grid(True)
plt.show()

plt.show()
