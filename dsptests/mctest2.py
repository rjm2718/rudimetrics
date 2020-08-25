#!/usr/local/bin/python3

import scipy
import scipy.io.wavfile

import dnotes
from dsputils import *

import random
import sys

#n = dnotes.ArcheTypeNote()
#print(n)


# given mic check of N hits same note:
#  - find sections of sample to frame N notes
#  - normalize volumes, no clipping
#  - filter between 100hz and 8khz
#  - 512-pt FT with hanning window
#  - agreement on archetypal note length
#
#  sanity check that spectrogram isn't drastically different between instances
#  (avg freq bins, check deviation, discard outliers or reject entire mic check)
#
#  Algorithm 1:
#   - at each FT window, find most discriminating frequency bins: most power difference between adjacent bins
#   - across all windows, find most discriminating frequency bins


# given archetypal note that spans m windows:
#  given spectrogram from input, take that frame of m and normalize values,
#  then fuzzy compare to archetype



# read input file, locate note boundaries:
#  - spectrogram stream, find onset (percussive energy) and decay, 250ms max



# test 1: take 2 manually selected notes, build archetype note, then test

files = ['samples/pad1.wav', 'samples/pad2.wav', 'samples/pad3.wav', 'samples/pad4.wav']

M = 256  # window size
spm = 32 # new overlapping window this many samples

notes = []

for fn in files:

    Fs, s = scipy.io.wavfile.read(fn)
    print('{} {}'.format(max(s), len(s)))

    notes.append(spectrogram(s, M, spm))



# average notes to get prototype note?
notes = np.asarray(notes)
print(notes.shape)
an = np.zeros(notes[0].shape)
for n in notes:
    an += n

normalize(an)
print(an.shape)



# test distances from original notes to an

d0 = []
dt = []

dmn = 9e9
for note in notes:
    d = distance(an, note)
    dmn = min(dmn, d)
    d0.append(d)

for fn in ['samples/ft1.wav', 'samples/sn1.wav']:
    Fs, s = scipy.io.wavfile.read(fn)
    note = spectrogram(s, M, spm)
    print(len(note))
    dt.append(distance(an, note))

d0 = [x/dmn for x in d0]
dt = [x/dmn for x in dt]

print('train:')
print('\n'.join([str(x) for x in d0]))
print('test:')
print('\n'.join([str(x) for x in dt]))
print('')
rx = (sum(dt)/len(dt)) / (sum(d0)/len(d0))
print('ratio of avgs:', str(rx))
print('')
print('')



# ----------------------------------------------------

# input larger file, check every window
# Slide spectrogram (an.shape[0]) and check against an


fn = 'samples/pad-loud-soft.wav'
Fs, s = scipy.io.wavfile.read(fn)
s = s / 2.**15  # to float
for i in range(0, len(s)):
    s[i] += (random.random() - 0.5)/500.0

v = [0] * len(s) # detections

wlen = an.shape[0] * spm + M

inote = []
i = 0 # starting sample
while i < len(s)-wlen:
    c = s[i:i+wlen]
    note = spectrogram(c, M, spm)
    d = distance(an, note)
    if d < 5:
        print('i={} d={}'.format(i, d))
    v[i] = d
    i += spm





#sys.exit(0)

###

import matplotlib.pyplot as plt

plt.figure(1)

plt.subplot(411)
plt.plot(s)

plt.subplot(412)
plt.plot(v)

#binsT = np.asarray(notes[0]).T
##binsT[20] *= 1000
#plt.subplot(411)
#plt.imshow(binsT, cmap='hot', interpolation='nearest', aspect='auto')
#
#binsT = an.T
#plt.subplot(412)
#plt.imshow(binsT, cmap='hot', interpolation='nearest', aspect='auto')

plt.show()
