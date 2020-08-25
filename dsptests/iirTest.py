#!/usr/bin/python3

# pod: percussive onset detection 
#
# pod signature: normalized levels of all bins to identify type (at just that moment), and
#  can have approxmiate comparisons to other signatures.
#
#
# read mic-check file:
#
#  1. use pod and peak detection to find all percussive events
#  2. given signatures of all events, find the ones that look like the expect sequence 
#     for mic check.
#  3. test is keep trying until you have N events or close (as expected)
#
#
# If window is too big, then use results of above to design tight filters, do peak detection
# again using filters to get 1ms resolution & accuracy.
#
# Make test wav file with weak drum hits and lots of noise (percussive and harmonic) and use
# that for final test.

#
# 22Khz sample rate => 87hz bins @ N=256, 11.5ms resolution
#

import numpy
from numpy.core.multiarray import arange
import scipy
from scipy import signal
import scipy.io.wavfile
import matplotlib.pyplot as plt

from scipy.fftpack import fft


## {{{ ###############################################################################

def plotSpectr(s, Fs, color='g', win=numpy.hamming):

    N = len(s)
    s_ = s[:]
    s_ = s_ * win(N)
    T = N / float(Fs)
    freqs = arange(N/4) / T
    Y = fft(s_) / N
    Y = Y[range(N/4)]

    plt.plot(freqs, abs(Y), color)


# 2nd order 11khz lowpass for 44100 sample rate
# http://www-users.cs.york.ac.uk/~fisher/mkfilter/trad.html
def iir_11khz(s):
    GAIN = 3.426409080e+00
    xv=[0,0,0]
    yv=[0,0,0]
    N = len(s)
    for n in xrange(0, N):
        xv[0] = xv[1]; xv[1] = xv[2]
        xv[2] = s[n] / GAIN
        yv[0] = yv[1]; yv[1] = yv[2];
        yv[2] =   (xv[0] + xv[2]) + 2 * xv[1] + ( -0.1715759537 * yv[0]) + (  0.0041730234 * yv[1])
        s[n] = yv[2]


# should be equivalent to scipy.signal.lfilter
# gain is a[0]
def iir(s, b, a):

    xv = [0.] * len(b) # numerator state
    yv = [0.] * len(a) # denominator state

    M = len(b) - 1
    N = len(a) - 1

    for n in xrange(0, len(s)):

        for i in range(0, M):
            xv[i] = xv[i+1]
        for i in range(0, N):
            yv[i] = yv[i+1]

        xv[M] = s[n] / a[0]
        yv[N] = 0.

        for i in range(0, M+1):
            yv[N] += xv[i] * b[i]
        for i in range(0, N):
            yv[N] += yv[i] * a[i+1]

        s[n] = yv[N]
    

## }}} ###############################################################################


plt.figure(1)

#fn = 'labeled/mic-check-1.wav'
fn = 'samples/mic-check-short.wav'
Fs, s = scipy.io.wavfile.read(fn)
s = s / 2.**15




s = s.tolist()


iir_11khz(s)
s = s[0::2]
Fs /= 2
N = len(s)



## test add sine
#ns = numpy.arange(N)
#ws = ns * 2 * numpy.pi * 7000 / Fs
#s += numpy.sin(ws)/50.0

plt.subplot(311)
plt.plot(s)


plt.subplot(312)
plotSpectr(s, Fs)
print '%s: Fs=%d len=%d' % (fn,Fs,N)

M = 256 # stft len
i = 0 # starting sample
bins = [] # energy in freq bins
while i < N-M:
    c = s[i:i+M]
    c = c * numpy.hamming(M)
    Y = abs(fft(c)).tolist()[0:M/2]
    bins.append(Y)
    i += M

# transpose for proper imshow
bins = numpy.asarray(bins).T.tolist()
plt.subplot(313)
plt.imshow(bins, cmap='hot', interpolation='nearest', aspect='auto')

plt.show()

