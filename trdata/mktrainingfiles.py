#!/usr/bin/python3

"""
Program to generate training/test data.  Query sqlite3 registry of samples and backgrounds, load files from
repo, apply various transformations to samples, then mix a random combination of everything.  Outputs a wav
file with at least a few samples (positive/negative drum/!drum note examples) per sec on top of random
backgrounds.  Outputs a labels file with an offset,label pair on each line indicating where the samples are
in the wave file.
"""

import math
import random
import traceback
import os.path
import sqlite3
import argparse
import numpy as np
import scipy.io.wavfile

from dsp import utils as u

# all paths to sound files are relative to REPO_BASE
home = os.path.expanduser("~")
REPO_BASE = os.path.join(home, 'rudimetrics', 'trdata', 'repo')
REPO_DB = os.path.join(REPO_BASE, 'repo.db')

# globals
SR = 0  # sample rate


class Sample:
    def __init__(self, fn, data, labels=[], apply_effects=False):
        self.fn = fn
        self.data = data
        self.labels = labels.split(',')
        self.apply_effects = apply_effects

    def __len__(self):
        return self.data.shape[0]

    def __str__(self):
        return self.fn


class SampleWrite:
    def __init__(self, sample, t_start):
        self.sample = sample;
        self.t_start = t_start
        self.offset_written = 0

    def __len__(self):
        return len(self.sample)

    def __str__(self):
        return '{}  @{}'.format(self.sample.fn, self.t_start)


def triangle_sine_sweep(N):
    a = np.zeros(N)

    # sweep these frequencies, up then back down
    f0 = 0 * 2 * np.pi
    f1 = 1000 * 2 * np.pi
    phi = 0  # phase
    f = f0
    phi_delta = f / SR
    f_delta = (f1 - f0) / N

    for i in range(0, int(N / 2)):
        a[i] = math.sin(phi)
        phi += phi_delta
        f += f_delta
        phi_delta = f / SR
    for i in range(int(N / 2), N):
        a[i] = math.sin(phi)
        phi += phi_delta
        f -= f_delta
        phi_delta = f / SR

    return a


def synthesize_backgrounds():
    """Create a variety of 10 second background samples: variations with white noise, tone sweeps."""

    backgrounds = []
    T = 5
    Ns = SR * T

    noise = np.array([2 * random.random() - 1.0 for _ in range(Ns)])
    noise8 = noise * 0.08 * 0.5
    noise2 = noise * 0.02 * 0.5
    backgrounds.append(Sample('noise8', noise8, 'bg'))
    backgrounds.append(Sample('noise5', noise2, 'bg'))

    tswp = triangle_sine_sweep(Ns) * 0.10 * 0.5
    backgrounds.append(Sample('sweep', tswp, 'bg'))

    backgrounds.append(Sample('sweep+noise', tswp + noise8, 'bg'))
    backgrounds.append(Sample('sweep-low+noise', tswp * 0.5 + noise2, 'bg'))

    return backgrounds


def noisyspike(z1=10):
    sw = random.randint(10, 40)
    d = [0.] * z1
    d.extend([.5] * 2)
    d.extend([-.5] * 2)
    d.extend([.99] * sw)
    d.extend([-.99] * sw)
    d.extend([.5] * int(sw / 4))
    d.extend([-.5] * int(sw / 4))
    d.extend([.25] * 5)
    d.extend([-.25] * 5)
    d.extend([.1] * 10)
    d.extend([-.1] * 10)
    d.extend([0] * (4096 - len(d)))
    d = list(map(lambda x: x + (random.random() - 0.5) / random.randint(5, 25), d))
    return np.asarray(d).astype('float32')


def generate_with_effects(originals):
    """
    For each sample, return new versions with various transformations applied:
    1. different volume levels (lower -- most original samples are already full range)
    2. add noise
    3. pitch adjust?
    4. add reverb, other distortions that a phone mic might add ... ?
    """

    samples = []

    for s in originals:
        noise = np.array([2 * random.random() - 1.0 for _ in range(len(s))]) * 0.7
        noise10 = noise * 0.10 * 0.5
        noise2 = noise * 0.02 * 0.5

        samples.append(Sample(s.fn + '+noise2', s.data + noise2, ','.join(s.labels)))
        samples.append(Sample(s.fn + '+noise10', s.data + noise10, ','.join(s.labels)))

        samples.append(Sample(s.fn + '+v50', s.data * 0.5, ','.join(s.labels)))
        samples.append(Sample(s.fn + '+v20', s.data * 0.2, ','.join(s.labels)))
        samples.append(Sample(s.fn + '+v2', s.data * 0.1, ','.join(s.labels)))

        samples.append(Sample(s.fn + '+v50+noise2', s.data * 0.5 + noise2, ','.join(s.labels)))
        samples.append(Sample(s.fn + '+v20+noise10', s.data * 0.2 + noise10, ','.join(s.labels)))
        samples.append(Sample(s.fn + '+v2+noise2', s.data * 0.1 + noise2, ','.join(s.labels)))

    return samples


def load_samples(backgrounds=False, label_filter=None):
    def resample_to_sr(sr_from, data, fn):
        return u.resample(sr_from, SR, data, fn)

    conn = sqlite3.connect(REPO_DB)  # default autocommit ... apparently not

    # load backgrounds ... no effects applied
    samples = []  # { fn,
    table_name = 'samples' if not backgrounds else 'backgrounds'

    c = conn.cursor()
    c.execute("SELECT filepath,labels,effects FROM {}".format(table_name))
    for row in c.fetchall():
        fn = row[0]
        labels = row[1]
        effects = row[2]
        if label_filter and not label_filter(labels):
            continue
        try:
            data = u.load_wav(fn, resample_to_sr)
            if data is not None:
                samples.append(Sample(fn, data, labels, effects))
            else:
                print('loading return None for', fn)

        except FileNotFoundError as f:
            print('failed to load background', fn)
            print('will skip file\n')
        except Exception as e:
            print('failed to load background', fn)
            print(e)
            traceback.print_exc()
            print('will skip file\n')

    c.close()

    return samples


def make_inverse_weights(v):
    """given list of values, return list of weights that sum to 1, where each value is
    mapped to a weight in inverse proportion.  Small values are adjusted to be no less than
    5x smaller than average value from list."""
    v = list(v)
    avg = sum(v) / len(v)
    weights = map(lambda w: max(w, avg / 5), v)
    weights = map(lambda w: avg / w, weights)  # inverse
    weights = list(weights)
    ws = sum(weights)
    weights = map(lambda w: w / ws, weights)  # make it sum to 1
    return list(weights)


def mix_and_record(mix_list, fn_wav, fn_labels, end_at):
    """mix_list needs to be in order of ascending offset"""

    with open(fn_labels, 'w') as outf_labels:

        buf = np.zeros(end_at)
        for m in mix_list:
            i = m.t_start
            j = min(end_at, i + len(m))
            l = j - i
            buf[i:j] += m.sample.data[0:l]
            #print(m)
            if 'bg' not in m.sample.labels:
                outf_labels.write('{}:{}\n'.format(i, ','.join(m.sample.labels)))

        scipy.io.wavfile.write(fn_wav, SR, buf)


def main():
    global SR

    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--sample-rate", dest="SR", help="sample rate", type=int, default=44100)
    ap.add_argument("-s", "--seconds", dest="DS", help="duration in seconds", type=int, default=300)
    ap.add_argument("-l", "--labels", dest="LABELS", help="comma-sep list of labels to include", type=str, required=False)
    ap.add_argument("-o", "--output", dest="OUTPATH",
                    help="output filename without extension, .wav and .labels will be created", type=str, required=True)
    args = ap.parse_args()

    SR = args.SR
    end = args.DS * SR

    lbl_filt = None
    if args.LABELS:
        inclbs = set(args.LABELS.split(','))
        def lbl_filt(labels_str):
            for l in labels_str.split(','):
                if l in inclbs:
                    return True
            return False

    samples = load_samples(label_filter=lbl_filt)
    generated_samples = generate_with_effects(filter(lambda s: s.apply_effects, samples))
    samples.extend(generated_samples)
    smp_weights = make_inverse_weights(map(len, samples))

    backgrounds = load_samples(backgrounds=True)
    for s in backgrounds:
        s.data = s.data * 0.5
    backgrounds.extend(synthesize_backgrounds())
    bg_weights = make_inverse_weights(map(len, backgrounds))

    print('')
    mix_list = []

    # write background tracks
    pos = 0
    while pos < end:
        s = np.random.choice(backgrounds, p=bg_weights)
        mix_list.append(SampleWrite(s, pos))
        pos += len(s)

    # write samples, with average spacing of 200ms
    avgSpacingSamples = SR * .3
    avgDistSamples = SR * .35
    minSampSpacingSec = 0.005
    minSampSpacing = int(minSampSpacingSec * SR)
    pos = minSampSpacing
    while pos < end:
        s = np.random.choice(samples, p=smp_weights)
        mix_list.append(SampleWrite(s, pos))

        rs = int(random.gauss(avgSpacingSamples, avgDistSamples))
        if rs < minSampSpacing:
            rs = random.randint(minSampSpacing, int(avgSpacingSamples / 5))
        pos += rs

    mix_list.sort(key=lambda s: s.t_start)

    print('average notes/sec will be ', len(mix_list) / args.DS)

    mix_and_record(mix_list, args.OUTPATH + '.wav', args.OUTPATH + '.labels', end)

    print('done!')


if __name__ == '__main__':
    main()
    # SR=8000
    # import matplotlib.pyplot as plt
    # a = triangle_sine_sweep(SR*100)
    # plt.plot(a)
    # plt.show()
