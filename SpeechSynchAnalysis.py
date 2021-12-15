# -*- coding: utf-8 -*-
"""
SPEECH SYNCHRONY ANALYSIS

Analysis of the Speech Synchrony Test as described in Lizcano et al. 2022.
The script perfoms the following steps:
    (1) Extracts the envelope of the produced speech signal and filters it around the stimulus syllabic rate.
    (2) Computes the PLVs between the produced and perceived filtered envelopes, in windows of 5 secs length, with an overlap of 2 secs
    (3) Averages the PLVs within each audio file (i.e. run1 and run2)
    (4) Control for consistency between runs
    (5) Gives the probability of the participant of being a high or low synchronizer.

Created on Wed Dec  8 13:32:12 2021

@author: Cecilia Mares 2021 ceciliap.maresr@gmail.com
"""

import funcs
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
get_ipython().magic('reset -sf') 


# Name of the audio file with the recorded speech
# subject_code = 'high'
subject_code = 'low'

# Version of the test used 
Test_Version = 'ExpAcc' # ExpAcc: for the Explicit Accelerated
# Test_Version = 'ImpFix' # ImpFix: for the Implicit Fixed


#############################################################################################################
# STEP 1: Loads the stimulus and both Runs of the subject's responses (signals)
#############################################################################################################
fs, y     = funcs.read_WAV('stimulus_acc.wav')
fs_h1, h1 = funcs.read_WAV(subject_code + '_run1.wav')
fs_h2, h2 = funcs.read_WAV(subject_code + '_run2.wav')

# Demeans and takes the envelope of the stimulus and of the spoken syllables
y_env  = funcs.gets_envelope(y)
h1_env = funcs.gets_envelope(h1)
h2_env = funcs.gets_envelope(h2)

# Applies a bandpass filter to the envelopes
bp_freqs = [3.3, 5.7] # cut frequencies
y_env_filt  = funcs.filt_bandpass(y_env,  fs,    bp_freqs)
h1_env_filt = funcs.filt_bandpass(h1_env, fs_h1, bp_freqs)
h2_env_filt = funcs.filt_bandpass(h2_env, fs_h2, bp_freqs)

# Resamples the filtered envelopes
fs_new = 100 # new sample frequency
y_env_filt_resamp  = funcs.resample(y_env_filt, fs, fs_new)
h1_env_filt_resamp = funcs.resample(h1_env_filt, fs_h1, fs_new)
h2_env_filt_resamp = funcs.resample(h2_env_filt, fs_h2, fs_new)

# Estimates the spectrum of the envelopes for visualization purposes.
xf1, yf1 = funcs.freqSpect(h1_env_filt_resamp, fs_new)
xf2, yf2 = funcs.freqSpect(h2_env_filt_resamp, fs_new)

#############################################################################################################
# STEP 2 & 3: Computes the PLV between the produced and perceived filtered envelopes
#############################################################################################################
theta_y  = funcs.angles(y_env_filt_resamp)
theta_h1 = funcs.angles(h1_env_filt_resamp)
theta_h2 = funcs.angles(h2_env_filt_resamp)

T = 5 # Size of window
shift = 2 # Size of overlap

plv1, time1 = funcs.PLVevol(theta_y, theta_h1, T, shift, fs_new)
plv2, time2 = funcs.PLVevol(theta_y, theta_h2, T, shift, fs_new)


#############################################################################################################
# Visualization of the data
#############################################################################################################
funcs.plotting(h1, fs_h1, h1_env_filt_resamp, fs_new, xf1, yf1, plv1, time1, h2, fs_h2, h2_env_filt_resamp, fs_new, xf2, yf2, plv2, time2)


#############################################################################################################
# STEP 4: Both PLVs should pass a control of consistency between each other
#############################################################################################################
flag_Exclude, plvs = funcs.Exclusion_Criteria(plv1, plv2, Test_Version)


#############################################################################################################
# STEP 5: If non of the exclusion criteria is reached, the probability of being High (1- prob of Low) is computed
#############################################################################################################
if flag_Exclude == 0:
    funcs.Probability_High(plvs, Test_Version)


