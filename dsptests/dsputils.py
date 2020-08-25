import math
import numpy as np
from scipy.fftpack import fft


def spectrum(x):
   """returns real valued fft of x (array like), discarding mirrored half of spectrum"""

   M = len(x)
   # make window, element multiply
   x = x * np.hanning(M) #returns ndarray shape (M,)
   Y = np.abs(fft(x))
   return Y[0 : int(M/2 + 1)]
   
# spectrogram, normalized
def spectrogram(s, M=256, spm=32, maxbins=128):

    i = 0 # starting sample
    bins = [] # energy in freq bins
    while i < len(s)-M and len(bins) < maxbins:
        c = s[i:i+M]
        Y = spectrum(c)
        ampByFreq(Y)
        bins.append(Y)
        i += spm
   
    normalize(bins)
    note = np.asarray(bins)
    return note
   

# there's some cleaner numpy way to do this in 1 line
def normalize(Y):
   """scale all values in 2-d Y wrt 1."""

   mx = -1.
   for c in Y:
       for r in c:
           mx = max(mx, abs(r))

   for i in range(len(Y)):
       c = Y[i]
       for j in range(len(c)):
           Y[i][j] /= mx


def ampByFreq(Y):
    for i in range(len(Y)):
        Y[i] *= (math.pow(i, 1.2) +1)


def distance(n1, n2):

    sq = np.power(n1 - n2, 2)
    return math.sqrt(np.sum(sq))
