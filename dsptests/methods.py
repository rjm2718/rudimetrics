
# from ipython: %run scratch.py

# plt.plot(snd, 'r')



#import scipy
#
#fn = 'ls1.wav'
#
#sampFreq, snd = scipy.io.wavfile.read(fn)
#
#print 'snd < fn'
#print ' type=', snd.dtype
#print ' channels=', len(snd.shape)
#print ' sampFreq=', sampFreq
#print '', len(snd),'samples, %.3f seconds'%(float(len(snd))/sampFreq)



def plotSpectr(s, Fs):

    import matplotlib.pyplot as plt
    from numpy.core.multiarray import arange
    from scipy.fftpack import fft

    N = len(s)
    T = N / float(Fs)
    freqs = arange(N/2) / T


    Y = fft(s) / N
    Y = Y[range(int(N/2))]

    plt.plot(freqs, abs(Y))


def foo():
    print('nf3')
