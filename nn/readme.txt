Oct 6, 2020 notes/plan
---------------------------------

Problem statement: percussive detection is hard because of background noise and overlapping notes.  note detection is
hard for percussive notes because of noisy spectrum.  We want to be able to detect and time drum, hhat, pad notes that
are closely spaced (10s of millisec apart, and overlapping), low in volume against background noise, and may be distored
or clipped by a cheap smartphone microphone.  Background noise can include sounds with a percussive element, such as a
hammered piano string and random pops and bangs.

Ideas to test:
> train nn on samples that begin slightly before their percussive onset, so the nn learns the correct part of the note.

> using a GAN, after detection, subtract out the approximate note from the audio signal, see if that allows better
detection of overlapping notes.  We may give up and train the nn to detect flams directly.

> move to 11025 sample rate.  quickly iterate on spectrogram and nn hyperparams.  program to automate this or look into
tf framework for support.

> clean up samples.  have easy/med/hard versions out of mktrainingfiles.py.  fade in/out bg samples.  We need more
diversity of training data (especially pad hits which are usually low volume), and be able to do lots more training
and experimentation.

> unsupervised learning of training sample categories ... this might help a lot because we are interested in accurate
discrimination more than identification of notes.

> far future ... state estimation.  Based on drum pattern ... detected or prescribed, train a Markov or Bayesian
network on expected detections.  I don't suspect a rnn will be successful.

> incorporate dsp techniques into system, some of which can be inputs to a further trained network.  POD, auto-correlation
from archetype notes.  If the user tells us what we need to detect, we can apply stringent filters.  With real samples
from user, do some few-shot learning), plus given the pretrained-on-many-note-types-network explore which network elements and
outputs can be used to best discriminate note to detect (we care more about timing and discrimination than perfect
identification).

> research best ML practices deal with detection of overlapping speakers