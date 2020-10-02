#!/usr/bin/python3

"""
Open .wav and .labels file, plot vertical lines at offset with a color to indicate class.
"""

import argparse
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from dsp import utils as u

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("WAVFILE", help="input wav file", type=str)
    ap.add_argument("LABELSFILE", help="input labels file", type=str)
    args = ap.parse_args()

    wavedata = u.load_wav(args.WAVFILE)

    label2offsets = {}
    for line in open(args.LABELSFILE):
        (offset, labels) = line.strip().split(':')
        offset = int(offset)
        for l in labels.split(','):
            offsets = label2offsets.get(l, [])
            offsets.append(offset)
            label2offsets[l] = offsets

    # subsample for sake of plotting
    subsamp = max(1, int(math.log(len(wavedata), 2)) - 17)

    wavedata = wavedata[0::subsamp]
    xd = np.arange(0, len(wavedata)) * subsamp

    # xd = np.arange(0, len(wavedata))

    fig, axes = plt.subplots()

    axes.plot(xd, wavedata, linewidth=1)

    for l in label2offsets:
        offsets = label2offsets[l]
        axes.scatter(offsets, np.ones(len(offsets)), marker='v', label=l)

    axes.legend()
    fig.tight_layout()
    plt.margins(0, 0)
    plt.show()


if __name__ == '__main__':
    main()
