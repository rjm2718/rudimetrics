#!/usr/bin/python3
import random
import traceback
import os.path
import sqlite3
import argparse
import numpy as np
import scipy.io.wavfile

from dsp import utils as u

# inputs: list of labels for positives and negatives, effects to apply, density, length, % samples for training/test
# output: wav file + labels file,  labels file contains (labels,offset) on each line; file set each for training/test

# sample rate: this should be an option, we will be playing with it later ... or maybe jsut stick with 44.1khz until
# we need more cpu

# given samples from query to repo:
#  - convert sample rate and mono as needed

# all paths to sound files are relative to REPO_BASE
home=os.path.expanduser("~")
REPO_BASE = os.path.join(home, 'rudimetrics', 'trdata', 'repo')
REPO_DB = os.path.join(REPO_BASE, 'repo.db')

# globals
SR = 0  # sample rate

class Sample:
    def __init__(self, fn, data, labels=[], apply_effects=False):
        self.fn = fn
        self.data = data
        self.labels = labels
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


def synthesize_backgrounds():
    return []

def generate_with_effects(originals):
    return []

def load_samples(backgrounds=False):

    def resample_to_sr(sr_from, data, fn):
        return u.resample(sr_from, SR, data, fn)

    conn = sqlite3.connect(REPO_DB) # default autocommit ... apparently not

    # load backgrounds ... no effects applied
    samples = [] # { fn,
    table_name = 'samples' if not backgrounds else 'backgrounds'

    c = conn.cursor()
    c.execute("SELECT filepath,labels,effects FROM {}".format(table_name))
    for row in c.fetchall():
        fn = row[0]
        labels = row[1]
        effects = row[2]
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
    weights = map(lambda w: avg / w, weights) # inverse
    weights = list(weights)
    ws = sum(weights)
    weights = map(lambda w: w / ws, weights)  # make it sum to 1
    return list(weights)


def mix_and_record(mix_list, fn_wav, fn_labels, end_at):
    """mix_list needs to be in order of ascending offset"""
    # outf_labels = open(fn_labels, 'w')

    buf = np.zeros(end_at)
    for m in mix_list:
        i = m.t_start
        j = min(end_at, i + len(m))
        l = j - i
        buf[i:j] += m.sample.data[0:l]

    scipy.io.wavfile.write(fn_wav, SR, buf)


    # BS=4096
    # buf = np.zeros(BS)
    #
    # m0 = 0  # index in mix_list of first SampleWrite that isn't completely written out yet
    # i = 0  # current offset pointer
    # while i < end_at:
    #     j = min(i + BS, end_at) # so range for this write is i to j
    #     for m in range(m0, len(mix_list)):
    #         sw = mix_list[m]
    #         if sw.offset_written >= len(sw.sample): # then we're done writing this sw
    #             m0 += 1
    #             continue
    #         if sw.t_start >= j: # then this and rest won't be used yet
    #             break
    #
    #         # sw will go into this buffer ... find out where and how much of it
    #         # from/to offsets in buf
    #         buf_offset = sw.t_start - i
    #         n = min(BS - buf_offset, len(sw.sample)) # number of samples to write
    #         buf_offset_to = buf_offset + n
    #
    #         # from/to offsets in sw.sample.data
    #         smp_offset = sw.offset_written
    #         smp_offset_to = smp_offset + n
    #         buf[buf_offset:buf_offset_to] += sw.sample.data[smp_offset:smp_offset_to]
    #
    #         sw.offset_written += n
    #
    #     #
    #     i += 1
    #
    # outf_labels.close()
    # outf_wav.close()


def main():

    global SR

    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--sample-rate", dest="SR", help="sample rate", type=int, default=44100)
    ap.add_argument("-s", "--seconds", dest="DS", help="duration in seconds", type=int, default=300)
    ap.add_argument("-o", "--output", dest="OUTPATH", help="output filename without extension, .wav and .labels will be created", type=str, required=True)
    args = ap.parse_args()

    SR = args.SR
    end = args.DS * SR

    samples = load_samples()
    generated_samples = generate_with_effects(filter(lambda s: s.apply_effects, samples))
    samples.extend(generated_samples)
    smp_weights = make_inverse_weights(map(len, samples))

    backgrounds = load_samples(backgrounds=True)
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
    avgSpacingSamples = SR * .2
    avgDistSamples = SR * .2
    minSampSpacingSec = 0.005
    minSampSpacing = int(minSampSpacingSec * SR)
    pos = minSampSpacing
    while pos < end:
        s = np.random.choice(samples, p=smp_weights)
        mix_list.append(SampleWrite(s, pos))

        rs = int(random.gauss(avgSpacingSamples, avgDistSamples))
        if rs < minSampSpacing:
            rs = random.randint(minSampSpacing, int(avgSpacingSamples/10))
        pos += rs

    mix_list.sort(key=lambda s: s.t_start)

    print('average notes/sec will be ', len(mix_list)/args.DS)

    mix_and_record(mix_list, args.OUTPATH + '.wav', args.OUTPATH + '.labels', end)

    print('done!')

if __name__ == '__main__':
    main()
