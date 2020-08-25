#!/usr/bin/python3

# notes:
#
#  To discriminate between sticks ... that's going to be hard.
#
#
#  general peaks with my 2 sticks hit together:
#   439, 1104, 1936, 2912, 4126 (lower), 5424 (peak of peaks), 6845, 8097, 9983
#
#  general peaks with left stick on pad:
#   47, 64, 103, 122, 194, 300, 434,448, 605, 1106, 1945, 2925
#  general peaks with right stick on pad:
#   46, 64, 104, 123, 194, 300, 435,448, 605, 1110, 1908, 2906
#  from many hits on pad:
#   68, 123, 214, 325, 434, 654, 1118, 2913, 4050, 5360, 6808
#
#  pad fundamental is clearly at 123; stick fundamental at 435
#   > 1105, 2912 also common good choice,
#
#  Since we're looking at high frequencies (relative to sample length), windowing is
#  less important.  But Hamming is easy enough.

# TODO
#  1. generate 1khz sine in 4096 record
#  2. window it
#  3. apply what you think is a narrow bandpass with such-and-such characteristics
#  4. confirm
#



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

Fs = 44100
nyq_rate = Fs / 2.
N = 4096

# test signal
n = numpy.arange(N)

fi = 440
w = n * 2 * numpy.pi * fi / Fs
s = numpy.sin(w)

fi = 800
w = n * 2 * numpy.pi * fi / Fs
s += numpy.sin(w)

fi = 1106
w = n * 2 * numpy.pi * fi / Fs
s += numpy.sin(w)


s *= 10000



taps = firwin(201, [440,450, 1100, 1112], pass_zero=False, width=10, nyq=nyq_rate, window='hamming')

#taps = taps/float(sum(taps))


#plt.figure(1)
#plt.plot(arange(0, len(s)), s, 'b')

plt.figure(2)
plt.clf()
w, h = freqz(s, worN=N)
plt.plot((w[0:1024]/numpy.pi)*nyq_rate, numpy.absolute(h[0:1024]), linewidth=2)
plt.grid(True)

#

s2 = signal.convolve(s, taps, mode='same')

#plt.figure(3)
#plt.plot(arange(0, len(s2)), s2, 'b')

plt.figure(4)
plt.clf()
w, h = freqz(s2, worN=N)
plt.plot((w[0:1024]/numpy.pi)*nyq_rate, numpy.absolute(h[0:1024]), linewidth=2)
plt.grid(True)

plt.show()
