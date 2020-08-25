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
#
#
# build-up & resonance of a single hit may typically be around 50ms.  A 4096 record
# by contrast is about 93ms.
#
# TODO
#  Don't optimize early!  Make it work, then worry about cpu.
#  Definitely will have dynamic thresholding and looking for patterns in decay/build-up.
#  Extended decay of a loud hit can dwarf softer shorter hits.
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

fn = 'roll1.test.wav'
Fs, s = scipy.io.wavfile.read(fn)

N = len(s)
print 'samples = ', N, ' Fs=', Fs
nyq_rate = Fs / 2.

Nr = 2048



taps = firwin(301, [430,440], pass_zero=False, width=5, nyq=nyq_rate, window='hamming')

#taps = taps/float(sum(taps))


plt.figure(1)
plt.plot(arange(0, len(s)), s, 'b')

plt.figure(2)
plt.clf()
w, h = freqz(s, worN=Nr)
plt.plot((w[0:256]/numpy.pi)*nyq_rate, numpy.absolute(h[0:256]), linewidth=2)
plt.grid(True)

#

s2 = signal.convolve(s, taps, mode='same')

plt.figure(3)
plt.plot(arange(0, len(s2)), s2, 'b')

plt.figure(4)
plt.clf()
w, h = freqz(s2, worN=Nr)
plt.plot((w[0:256]/numpy.pi)*nyq_rate, numpy.absolute(h[0:256]), linewidth=2)
plt.grid(True)

plt.show()
