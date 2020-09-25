import sys
import numpy as np
import scipy
from scipy.fftpack import fft
from scipy import signal
import scipy.io
import scipy.io.wavfile
import wavio


def spectrum(x):
    """returns real valued fft of x (array like), discarding mirrored half of spectrum. Hann window applied."""

    M = len(x)
    # make window, element multiply
    x = x * np.hanning(M)  # returns ndarray shape (M,)
    Y = np.abs(fft(x))
    return Y[0: int(M / 2 + 1)]


#
def spectrogram(s, M=256, spm=32):
    """ s is sample data, M is window size, spm is stride
        (4096,) input, returns ((4096-256)/spm, 129) -> (121, 129)
        """

    i = 0  # starting sample
    bins = []  # energy in freq bins
    while i <= len(s) - M:
        c = s[i:i + M]
        Y = spectrum(c)
        bins.append(Y)
        i += spm

    bins = np.asarray(bins)
    bins = normalize(bins)
    return bins

def resample(sr_from, sr_to, data, fn):
    """resample, returned array length will be scaled by sr_to/sr_from.  Hann window applied for samples 8192 or smaller
    in size.
    """
    if sr_from==sr_to: return data

    Nf = data.shape[0]
    Nt = int(Nf * sr_to / sr_from)

    # TODO not sure what the best strategy is
    window = None
    if Nf <= 8192:
        window = 'hann'
        # window = np.ones(Nf))

    #print('resamping {}, {} -> {}'.format(fn, sr_from, sr_to))
    return signal.resample(data, Nt, window=window)


def normalize(b):
    mx = np.abs(b).max()
    if mx < 0.000000001:
        return b
    return b / mx


def load_spectrogram(fn):
    data = load_wav(fn)
    return spectrogram(data)


def convert_samplerate(data, sr_from, sr_to):
    if sr_from != sr_to:
        print('warning: sample rate conversion {} to {} not implemented yet'.format(sr_from, sr_to))
    return data


def load_wav(fn, sample_rate_conv_func=None):
    """Load wav file at path fn.  Should be able to read multiple wav formats.
    Will mix to mono as needed.  Returns float32 numpy array (values between -1 and 1).

    If sample_rate_conv_func is given, it will be invoked as sample_rate_conv_func(sr, data, fn),
    where sr is sample rate read from file.  This gives the caller control of resampling
    so the best windowing can be applied.
    """

    # try both libs as needed to load wav file,
    try:
        sr, data = scipy.io.wavfile.read(fn)
        if not data.flags['WRITEABLE']:
            data = np.array(data)

        if data.dtype == 'int16':
            data = np.divide(data, 2 ** 15).astype('float32')

    # there's some expected exception here for some files that scipy.io.wavefile.read can't handle,
    # but I can't remember what
    except Exception as e:

        # wavio.read can't read floating point format
        w = wavio.read(fn)
        data = w.data
        if not data.flags['WRITEABLE']:
            data = np.array(data)

        if data.dtype == 'int32':
            data = np.divide(data, 2 ** (w.sampwidth * 8 - 1)).astype('float32')

        sr = w.rate

    if data is None or len(data)==0:
        print('warning: load_wav has no data loaded for', fn)
        return data

    mx = np.abs(data).max()
    if mx < 0.000000001:
        print("warn: all zeros in", fn)

    data = convert_to_mono(data)

    if np.isnan(np.max(data)):
        raise Exception("np.nan found in data")

    if sample_rate_conv_func:
        data = sample_rate_conv_func(sr, data, fn)
        if data is None:
            print('warning: sample_rate_conv_func returned nothing')

    return data


def convert_to_mono(data):
    """Mix 2 or more channels down to 1, returns reshaped data."""

    if data is None or len(data) == 0:
        return data

    if len(data.shape) == 1:
        return data

    nc = data.shape[1]

    for i in range(1, data.shape[1]):
        data[..., 0] += data[..., i]

    data = data[..., 0]
    return np.divide(data, nc)
