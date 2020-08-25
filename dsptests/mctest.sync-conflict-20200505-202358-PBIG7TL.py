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
# TODO: find expected decay from mic check, so we can debounce plus detect two hits close together.
#
# Make test wav file with weak drum hits and lots of noise (percussive and harmonic) and use
# that for final test.
#
#
# 1208 bpm -- world record drummer! (20ms).  But shortest interval psychologist test subject
#  can detect is more like 50ms.

# TODO conv-net siamese network !

#
# 22Khz sample rate => 87hz bins @ N=256, 11.5ms resolution
#

import math
import random
import numpy as np
from numpy.core.multiarray import arange
import scipy
from scipy import signal
import scipy.io.wavfile
import matplotlib.pyplot as plt

from scipy.fftpack import fft

D = False

## {{{ ###############################################################################

def plotSpectr(s, Fs, color='g', win=np.hamming):

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
    for n in range(0, N):
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

    for n in range(0, len(s)):

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
    

def logDiff(S, S1): # S current, S1 previous
    X = [0] * len(S)
    for i in range(0, len(S)):

        d = S[i] - S1[i]

        # TODO normalize levels during mic check (though mic check could certainly be unreliable)
        # (or, assuming we have auto-gain-control, we can keep this value fixed)
        d *= 2 # tune for desired region in log10 curve

        if d > 0:
            d += 0.99
        if d < 0:
            d -= 0.99

        if (d > 1):
            X[i] = 20*math.log10(d)
        elif (d < -1):
            d = -d
            X[i] = -20*math.log10(d)
        else:
            X[i] = 0.0

    return X



T = 9
def pef(X):
    e = 0
    for x in X:
        if x > T:
            e += 1
    #    if x > T*2:
    #        e += 1
    return e

## }}} ###############################################################################


plt.figure(1)

#fn = 'labeled/mic-check-1.wav'
#fn = 'roll1.test.wav'

fn = 'samples/pad-loud-soft.wav'
#fn = 'samples/mic-check-short.wav'
#fn = 'mic-check-m2.wav'
#fn = 'notes.wav'
Fs, s = scipy.io.wavfile.read(fn)
s = s / 2.**15  # to float

s = s.tolist()


# downsample
iir_11khz(s)
s = s[0::2]
Fs /= 2
N = len(s)

for i in range(0, N):
    s[i] += (random.random() - 0.5)/50.0


## test: add sine
#for i in range(0, N):
#    r = i * 2 * np.pi * 2500 / Fs
#    s[i] += math.sin(r) / 200.0

## test: add sine
#ns = np.arange(N)
#ws = ns * 2 * np.pi * 7000 / Fs
#s += np.sin(ws)/50.0

## test: TODO: add about 100 sines ... really flood the spectrum with non-percussive stuff

## flood signal with piano notes (a bit percussive) in the same freq range



plt.subplot(411)
plt.plot(s)


#plt.subplot(512)
#plotSpectr(s, Fs)
print('%s: Fs=%d len=%d' % (fn,Fs,N))



# filters:
#  pad
#   120hz  1
#   224hz  .9
#   346hz  .8
#   600hz  .7
#   820hz  .6
#
#  sticks
#   450hz  .5
#   1112hz .8
#   1952hz .9
#   2932hz 1
#   5500hz .9
#
# decay:
#  pad
#   0ms   .2
#   4ms   1
#  14ms   .6
#  30ms   .45
#  44ms   .3
#  80ms   .1
#
# sticks
#  0ms    1
#  2ms    .8
#  6ms    .3
#  12ms   .2
#  50ms   .1

# stft len (256: @ Fs=22k -> 11.6ms per stft bin)
# Need to experiment with this parameter to get best results; if we need more resolution
# then we go to narrow band filtering + peak detection
# NOTE: say you're doing 64th notes at 60bpm (that's super fast -- 16hz), then 11.6 resolution
#       is 27% of that total period.  Of course at that rate the drummer won't care as much
#       about accuracy.  Just measuring hits at that tempo is still a good project.  That said
#       I think 11.6ms is too much -- maybe half that at most.
# NOTE: instead of going to filtering for more resolution, don't forget you can do overlapping
#       windows (at the cost of cpu)  FYI all the little FFTs are adding up to a lot of cpu time
#       in Python ... do investigate ... yeah, ok, on our mobile devices doing native fft ops
#       it should take far less than 1ms to do 512 point FFTs.
M = 256

i = 0 # starting sample
bins = [] # energy in freq bins
while i < N-M:
    c = s[i:i+M]
    c = c * np.hamming(M)
    Y = abs(fft(c)).tolist()[0 : int(M/2)]
    bins.append(Y)
    i += M

sys.exit(0)
# transpose for proper imshow
binsT = np.asarray(bins).T.tolist() * 4
plt.subplot(412)
plt.imshow(binsT, cmap='hot', interpolation='nearest', aspect='auto')


# measuring decay
idecay = 0

# max peak
mpeak = 0



Se = []
Pe = [] # percussive energy by bin
pc = 0
lkout_until = 0
for i in range(1, len(bins)):
    D = i>=472 and i<=487

    if i < lkout_until: pass

    S = bins[i]
    S1 = bins[i-1]  # previous stft
    X = logDiff(S, S1)
    #print '%3d:  sum=%5.2f  f=%4.3f' % (i, sum(S), sum(X))

    medLock = False
    if idecay > 0:
        medLock = (i - idecay) < 8 

    pT = 120 # this simple threashold is what we'll need to tune, hopefully dynamically
    #pe = pef(X)
    pe = sum(X)
    if pe > pT or medLock:
        print('%3d (%4d):  %4d' % (i, i*M/2, pe))
        #print str(i) + ',' + ','.join(map(str, S))

        if idecay==0:
            idecay = i  # start of event

        if abs(pe) > mpeak:
            mpeak = abs(pe)

    else:  # else not in event

        if idecay > 0:
            ddur = i - idecay
            print('* ddur=',ddur, ' mpeak=',mpeak)
            pc += 1

        idecay = 0
        mpeak = 0

    Pe.append(pe)

    se = sum(S)
    Se.append(se)

    if D:
        #print ['%3d' % (val*10) for val in X[0:32]]
        #print ['%.2f' % (val*10) for val in S[0:32]]
        #print ''
        #print ['%.2f' % (val*10) for val in S1[0:64]
        #print ''
        #dx = [10*(x[0]-x[1]) for x in zip(S, S1)]
        #print ['%.2f' % val for val in dx[0:64]]
        #print sum(map((lambda x: 1 if x > 0 else 0), dx))
        #print ''
        pass


print('pc=',pc)

# TODO subtract scaled expected decay (to reveal new hits buried)
#   >> this is required -- on drums there are percussive ups and downs within a single hit, and simple
#      thresholding will result in many duplicates; therefore you *have* to characterize decay and 
#      subtract it out.
#   >> The way to do that: during mic check, just look at the spectrum for a characteristic number of 
#      blocks (i.e. a single hit's spectrogram); then average that spectrogram for all mic-check hits,
#      normalizing the amplitudes.  You then have the archetype spectrogram, which you can then use
#      to subtract from blocks later (scaled to amplitude of local peaks).  Problem may be, after looking
#      at spectrograms of hits in mic check, each hit has a surprising amount of differences (but higher
#      resolution analysis may resolve this)

# TODO next characterize and label all detected peaks

plt.subplot(413)
plt.plot(Se)

plt.subplot(414)
plt.plot(Pe)

plt.show()
