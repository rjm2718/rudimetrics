#!/usr/bin/python3

import traceback
import os.path
import sqlite3
import argparse
import numpy as np

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

# load catalog of samples from repo

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

def load_samples(sr, backgrounds=False):

    def resample_to_sr(sr_from, data, fn):
        return u.resample(sr_from, sr, data, fn)

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
    
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--sample-rate", dest="SR", help="sample rate", type=int, default=44100)
    args = ap.parse_args()

    backgrounds = load_samples(args.SR, backgrounds=True)
    bg_weights = make_inverse_weights(map(len, backgrounds))

    samples = load_samples(args.SR)
    smp_weights = make_inverse_weights(map(len, samples))

    #print(list(map(lambda w: '%.3f' % w, bg_weights)))
    for _ in range(20):
        c = np.random.choice(backgrounds, p=bg_weights)
        print(c, len(c))

    print('')
    for _ in range(50):
        c = np.random.choice(samples, p=smp_weights)
        print(c, len(c))




if __name__ == '__main__':
    main()
