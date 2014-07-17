"""
   Modified files are originated from http://zacharydenton.com/generate-audio-with-python/

   In this script, our goal is to generate a composited sine wave aduio file in which there are 4 sine waves, 
   each two are synthesized and adopted in each channels (number of channl are 2 by default).

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

    parser = argparse.ArgumentParser(prog="mac_audiogen")
    parser.add_argument('-lt', '--lt', help="left top (freq, onsteps, allsteps)", 
			nargs=4, default=(1000, 3, 5), type=int)
    parser.add_argument('-ld', '--ld', help="left down (freq, onsteps, allsteps)", 
			nargs=4, default=(15000, 3, 5), type=int)
    parser.add_argument('-rt', '--rt', help="right top (freq, onsteps, allsteps)", 
			nargs=4, default=(1000,  15, 55), type=int)
    parser.add_argument('-rd', '--rd', help="left down (freq, onsteps, allsteps)", 
			nargs=4, default=(15000, 15, 55), type=int)
    parser.add_argument('-b', '--bits', help="Number of bits in each sample", choices=(16,), default=16, type=int)

    parser.add_argument('-s', '--secs', help="Duration of the wave in seconds.(default=10)", default=10, type=int)

    parser.add_argument('-r', '--rate', help="Sample rate in Hz", default=48000, type=int)

    parser.add_argument('-a', '--amplitude', help="Amplitude of the wave on a scale of 0.0-1.0.", default=0.5, type=float)

    #parser.add_argument('filename', help="The file to generate.")


    args = parser.parse_args()

    params = ( (( args.lt[0], args.lt[1], args.lt[2]),
		( args.ld[0], args.ld[1], args.ld[2]) ),
	       (( args.rt[0], args.rt[1], args.rt[2]),
		( args.rd[0], args.rd[1], args.rd[2]) )
    )
    nparams = 2

    secs = args.secs
    bits = args.bits
    rate = args.rate
    amp = args.amplitude
    fname = 'audio_default.wav'
    tmp_fname = 'tmp.out'    

    t_start = time.time()

    channels = ( ( sine_wave(frq_left,  rate, amp, on_left, all_left), 
		   sine_wave(frq_right, rate, amp, on_right, all_right) ) 
		 for ((frq_left, on_left, all_left), (frq_right, on_right, all_right)) in params  )

    samples = compute_samples(channels, rate*secs)

    write_wavefile(tmp_fname, samples, rate * secs, 2, bits / 8, rate)

    shutil.move(tmp_fname, fname)

    print "File '%s' generated, cost: %5.3f(s)" % ( fname, time.time()-t_start )


if __name__ == "__main__":
    main()
