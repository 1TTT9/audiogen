"""
   Modified files are originated from http://zacharydenton.com/generate-audio-with-python/




"""
import time
import sys
import wave
import math
import struct
import random
import argparse
from itertools import *


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)



def sine_wave(frequency=440.0, framerate=44100, amplitude=0.5, onswitch=1, switch=5, skip_frame=0):
    '''
    Generate a sine wave at a given frequency of infinite length.
    '''
    if amplitude > 1.0: amplitude = 1.0
    if amplitude < 0.0: amplitude = 0.0

    onSteps, allSteps = onswitch, switch
    c = 0
    for i in count(skip_frame):
	c = int( frequency * (float(i) / float(framerate)) )  % allSteps + 1
	if onSteps >= c:
	    sine = math.sin(2.0 * math.pi * float(frequency) * (float(i) / float(framerate)))
	else:
	    sine = 0

        yield float(amplitude) * sine

"""
def sine_wave(frequency=440.0, framerate=44100, amplitude=0.5, skip_frame=0):
    '''
    Generate a sine wave at a given frequency of infinite length.
    '''
    if amplitude > 1.0: amplitude = 1.0
    if amplitude < 0.0: amplitude = 0.0
    for i in count(skip_frame):
        sine = math.sin(2.0 * math.pi * float(frequency) * (float(i) / float(framerate)))
        yield float(amplitude) * sine
"""

def compute_samples(channels, nsamples=None):
    '''
    create a generator which computes the samples.

    essentially it creates a sequence of the sum of each function in the channel
    at each sample in the file for each channel.
    '''
    return islice(izip(*(imap(sum, izip(*channel)) for channel in channels)), nsamples)


def write_wavefile(filename, samples, nframes=-1, nchannels=2, sampwidth=2, framerate=44100, bufsize=2048):
    "Write samples to a wavefile."

    w = wave.open(filename, 'w')
    w.setparams( (nchannels, sampwidth, framerate, nframes, 'NONE', 'not compressed') )

    max_amplitude = float(int((2 ** (sampwidth * 8)) / 2) - 1)
    # split the samples into chunks (to reduce memory consumption and improve performance)
    """
    for chunk in grouper(bufsize, samples):
        frames = ''.join(''.join(struct.pack('h', int(max_amplitude * sample)) for sample in channels) for channels in chunk if channels is not None)
        w.writeframesraw(frames)
    """
    for chunk in grouper(bufsize, samples):
	frames = ''
	for channels in chunk:
	    s = ''
	    if channels is not None:
		for sample in channels:
		    s += struct.pack('h', int(max_amplitude * sample))
		frames += s
        w.writeframesraw(frames)

    w.close()

    return filename


def main():
    """
    test_[freq]_[sampling_rate].wav
     - test_1k_48k.wav cost: 4.515(s)
     - test_15k_48k.wav cost: 4.547(s)
     - test_1k_48k.wav cost: 4.547(s)  #after adding k_switch
    """
    t_start = time.time()
    fa  = 1000.0
    fb  = 1000.0
    a = 1.0
    secs = 15.0
    bits = 16
    rate = 48000
    fname = 'test_1_5.wav'

    ''' (frequency, sampling_rate, amplitude, on_in_k_switchs, k_switchs) '''
    params_channels = ( (fa, rate, a, 5, 5), 
			(fb, rate, a, 5, 5) )

    # each channel is defined by infinite functions which are added to produce a sample.
    channels = ( (sine_wave(freq, rate, amp, on_sw, sw),) for (freq, rate, amp, on_sw, sw) in params_channels  )
    #channels = ( ( sine_wave(fa, rate, a),), ( sine_wave(fa, rate, a),) )

    # convert the channel functions into waveforms
    samples = compute_samples(channels, rate*secs)

    write_wavefile(fname, samples, rate * secs, 2, bits / 8, rate)

    print fname, "cost: %5.3f(s)" % ( time.time()-t_start )


if __name__ == "__main__":
    main()
