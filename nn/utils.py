import math
import numpy as np
import scipy
from scipy.fftpack import fft
import wavio


def spectrum(x):
   """returns real valued fft of x (array like), discarding mirrored half of spectrum"""

   M = len(x)
   # make window, element multiply
   x = x * np.hanning(M) #returns ndarray shape (M,)
   Y = np.abs(fft(x))
   return Y[0 : int(M/2 + 1)]


#
def spectrogram(s, M=256, spm=32):
    """ s is sample data, M is window size, spm is stride
        (4096,) input, returns ((4096-256)/spm, 129) -> (121, 129)
        """

    i = 0 # starting sample
    bins = [] # energy in freq bins
    while i <= len(s)-M:
        c = s[i:i+M]
        Y = spectrum(c)
        bins.append(Y)
        i += spm

    bins = np.asarray(bins)
    bins = normalize(bins)
    return bins
    

def normalize(b):

    mx = np.abs(b).max()
    if mx < 0.000000001:
        return b
    return b/mx

def load_spectrogram(fn):

    data = load_wav(fn)
    return spectrogram(data)

def load_wav(fn):
    """Load wav file from given filename.  Should be able to read multiple wav formats.
    Returns float32 numpy array (values between -1 and 1)."""

    # try both libs as needed to load wav file,
    try:
        Fs, data = scipy.io.wavfile.read(fn)
        if Fs != 44100:
            print('warning:',fn, 'sample rate not 44100')
            return None

        if data.dtype == 'int16':
            data = np.divide(data, 2**15).astype('float32')

    except Exception as e:

        try:
            w = wavio.read(fn)
            if w.rate != 44100:
                print('warning:',fn, 'sample rate not 44100')
            data = w.data

            if data.dtype == 'int32':
                data = np.divide(data, 2**(w.sampwidth*8-1)).astype('float32')

        except Exception as e:
            print(e)
            return None


    mx = np.abs(data).max()
    if mx < 0.000000001:
        print("warn: all zeros in", fn)
    
    data = monoification(data)

    return data

def monoification(data):

    if len(data.shape) == 1:
        return data

    nc = data.shape[1]

    for i in range(1,data.shape[1]):
        data[...,0] += data[...,i]

    data = data[...,0]
    return np.divide(data, nc)


def validate_data(data):

    if np.isnan(np.max(data)):
        print("np.nan found in data ... aborting!")
        sys.exit()

