=== trdata ===

subprojects:

1. manual process of organizing sample files by label

2. script to write file names / labels into repo.db

3. main program to generate sample files:

  a) inputs: list of labels for positives and negatives, effects to apply, density, length, % samples for training/test
  x) output: wav file + labels file,  labels file contains (labels,offset) on each line; file set each for training/test


# plan:
#  Keep a directory of unadulterated samples (raw from professional recording) that can have effects added
#  later.  Then other samples from drum machines that may already have effects.  Then another directory
#  for all the "real" samples from drums outside the studio recorded by smartphone mics.
#  It's hard to find a single category for many samples, so register them multiple times as needed.  The objective
#  is to create a signature for each note, not find the best single category.


# background noise samples: https://www.ee.columbia.edu/~dpwe/sounds/noise/