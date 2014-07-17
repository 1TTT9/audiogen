"""
   Modified files are originated from http://zacharydenton.com/generate-audio-with-python/




"""
import shutil
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

    parser = argparse.ArgumentParser(prog="audiogen")
    parser.add_argument('-lt1', '--lt1', help="left top 1", default=1, type=int)
    parser.add_argument('-lt2', '--lt2', help="left top 2", default=1, type=int)
    parser.add_argument('-1d1', '--ld1', help="left down 1", default=1, type=int)
    parser.add_argument('-ld2', '--ld2', help="left down 2", default=1, type=int)
    parser.add_argument('-rt1', '--rt1', help="right top 1", default=1, type=int)
    parser.add_argument('-rt2', '--rt2', help="right top 2", default=1, type=int)
    parser.add_argument('-rd1', '--rd1', help="right down 1", default=1, type=int)
    parser.add_argument('-rd2', '--rd2', help="right down 2", default=1, type=int)
    parser.add_argument('-sec', '--sec', help="Duration of the wave in seconds.", default=10, type=int)
    #parser.add_argument('filename', help="The file to generate.")
    args = parser.parse_args()

    """
    test_[freq]_[sampling_rate].wav
     - test_1k_48k.wav cost: 4.515(s)
     - test_15k_48k.wav cost: 4.547(s)
     - test_1k_48k.wav cost: 4.547(s)  #after adding k_switch
    """
    t_start = time.time()
    fa1 = 1000
    fa2 = 15000
    fb1 = 1000
    fb2 = 15000
    a = 0.5
    secs = args.sec
    bits = 16
    rate = 48000
    fname = 'test_4ch_%d_%d_%s.wav' % (fa1, fa2, secs)

    ''' (frequency, sampling_rate, amplitude, on_in_k_switchs, k_switchs) '''
    # each channel is defined by infinite functions which are added to produce a sample.
    #channels = ( (sine_wave(freq, rate, amp, on_sw, sw),) for (freq, rate, amp, on_sw, sw) in params_channels  )
    channels = ( ( sine_wave(fa1, rate, a, args.lt1, args.lt2), sine_wave(fa2, rate, a, args.ld1, args.ld2) ), 
		 ( sine_wave(fb1, rate, a, args.rt1, args.rt2), sine_wave(fb2, rate, a, args.rd1, args.rd2) ) )

    # convert the channel functions into waveforms
    samples = compute_samples(channels, rate*secs)

    #write_wavefile(fname, samples, rate * secs, 2, bits / 8, rate)

    tmpfname = 'tmp.out'
    write_wavefile(tmpfname, samples)
    shutil.move(tmpfname, fname)

    print fname, "cost: %5.3f(s)" % ( time.time()-t_start )


if __name__ == "__main__":
    main()
