# -*- coding: utf-8 -*-
"""

SPEECH SYNCHRONY ANALYSIS FUNCTIONS

Analysis of the Speech Synchrony Test as described in Lizcano et al. 2022.
The description of each function is explained below.

Created on Wed Nov 24 12:46:01 2021

@author: Cecilia Mares 2021 ceciliap.maresr@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
import scipy.io.wavfile as wavfile
import pickle
from scipy.stats import norm
import tkinter


def read_WAV(file_name):
    '''
    

    Parameters
    ----------
    file_name : Name of WAV file

    Returns
    -------
    fs : Sample frequency of WAV file.
    y : Array with data in WAV file.

    '''
    fs, y = wavfile.read(file_name)  
    return fs, y


def gets_envelope(y):
    '''
    

    Parameters
    ----------
    y : Signal.

    Returns
    -------
    envelope : Envelope of signal.

    '''
    envelope = np.abs(hilbert(y))
    envelope = envelope - np.mean(envelope)
    return envelope


def filt_bandpass(y,fs_ent,bp_freqs):
    '''
    

    Parameters
    ----------
    y : Signal.
    fs_ent : Sample frequency of signal.
    bp_freqs : cut frequencies of bandpass filter.

    Returns
    -------
    y_filt : signal filtered.

    '''
    sos = signal.butter(5, bp_freqs, btype='bandpass', analog=False, output='sos',fs=fs_ent)
    y_filt = signal.sosfilt(sos, y)
    return y_filt


def resample(y, fs_old, fs_new):
    '''
    

    Parameters
    ----------
    y : Signal.
    fs_old : Old sample frequency.
    fs_new : New sample frequency.

    Returns
    -------
    y_resampled : Signal resampled.

    '''
    y_resampled = y[::int(fs_old/fs_new)]
    return y_resampled


def freqSpect(y, fs):
    '''
    

    Parameters
    ----------
    y : Signal.
    fs : Sample frequency of signal.

    Returns
    -------
    xf : Vector of frequencies.
    yf : Power of signal.

    '''
    y = y - np.mean(y)
    N = len(y)
    Ts = 1.0 / fs
    
    yf = fft(y)
    xf = fftfreq(N, Ts)[:N//2]
    
    index10 = np.where(xf <= 10)
    xf = xf[index10]
    yf = yf[index10]
    
    yf = np.power(np.abs(yf),2)
    yf = yf / max(yf)
    
    return xf, yf


def angles(y):
    '''
    

    Parameters
    ----------
    y : Signal.

    Returns
    -------
    theta : Angles' vector.

    '''
    yy = hilbert(y)
    theta = np.unwrap(np.angle(yy))
    return theta


def PLVevol(theta_stim, theta_sign, T, shift, fs):
    '''
    

    Parameters
    ----------
    theta_stim : Angles' vector of the stimulus.
    theta_sign : Angles' vector of the signal.
    T : Size of window.
    shift : Size of overlap.
    fs : Sample frequency.

    Returns
    -------
    plv : Phase Locking Value between stimulus and signal.
    time : Timing vector.

    '''
    tmp = min(len(theta_stim), len(theta_sign))
    phase_diff = theta_stim[0:tmp] - theta_sign[0:tmp]
    duration = tmp / fs
    
    nT = round(fs*T)
    nshift = round(fs*shift)
    n_ant = 0
    i = 0
    
    time = np.empty(int(duration * fs / nshift))
    plv = np.empty(int(duration * fs / nshift))

    
    while (n_ant + nT) <= len(phase_diff):
        plv[i] = np.abs(np.sum(np.exp(1j*phase_diff[n_ant : n_ant + nT])))/ nT
        time[i] = 0.5 * (n_ant + n_ant + nT) / fs
        n_ant += nshift
        i += 1

    if (n_ant + nT) > len(phase_diff):
        plv[i] = np.abs(np.sum(np.exp(1j*phase_diff[n_ant:]))) / (len(phase_diff) - n_ant)
        time[i] = 0.3 * (len(phase_diff) + n_ant) / fs
       
    return plv, time

    
def plotting(sg1, fs1, envelope1, fs_new1, xf1, yf1, plv1, time1, sg2, fs2, envelope2, fs_new2, xf2, yf2, plv2, time2):
    '''
    

    Parameters
    ----------
    sg1 : Raw signal1.
    fs1 : Sample frequency of raw signal1.
    envelope1 : Envelope of signal1.
    fs_new1 : New sample frequency of signal1.
    xf1 : Vector of frequencies (signal1).
    yf1 : Power of signal1.
    plv1 : PLV between stimulus and signal1.
    time1 : Timing vector1.
    
    sg2 : Raw signal2.
    fs2 : Sample frequency of raw signal2.
    envelope2 : Envelope of signal2.
    fs_new2 : New sample frequency of signal2.
    xf2 : Vector of frequencies (signal2).
    yf2 : Power of signal2.
    plv2 : PLV between stimulus and signal2.
    time2 : Timing vector2.

    Returns
    -------
    None.

    '''
    time_sg1 = np.linspace(0.0, len(sg1)/fs1, len(sg1), endpoint=False)
    time_envelope1 = np.linspace(0.0, len(envelope1)/fs_new1, len(envelope1), endpoint=False)
    time_sg2 = np.linspace(0.0, len(sg2)/fs2, len(sg2), endpoint=False)
    time_envelope2 = np.linspace(0.0, len(envelope2)/fs_new2, len(envelope2), endpoint=False)
    
    plvMean1 = np.mean(plv1)
    plvMean2 = np.mean(plv2)
    
    fig = plt.figure(figsize=(30, 20))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    bx1 = fig.add_subplot(234)
    bx2 = fig.add_subplot(235)
    bx3 = fig.add_subplot(236)
    
    
    ax1.set_title('Run 1: Speech signal plus envelope', fontsize=20, fontweight='bold')
    ax1.plot(time_sg1, sg1, color='black', label='signal')
    ax1.plot(time_envelope1, envelope1, color='red', label='envelope')
    ax1.set_xlim([10, 15])
    
    ax2.set_title('Run 1: Produced envelope spectrum', fontsize=20, fontweight='bold')
    ax2.plot(xf1, yf1, color='blue')
    ax2.set_xlabel('Frequency (Hz)', fontsize=15)
    ax2.set_ylabel('Power', fontsize=15)
    ax2.set_xlim([0, 10])
    
    ax3.set_title('Run 1: Mean PLV = ' + str("{:.2f}".format(plvMean1)), fontsize=20, fontweight='bold')
    ax3.plot(time1, plv1, 'ob')
    ax3.set_xlabel('Time (sec)', fontsize=15)
    ax3.set_ylabel('Speech Synchrony (PLV)', fontsize=15)
    ax3.set_ylim([0, 1])
    
    bx1.set_title('Run 2: Speech signal plus envelope', fontsize=20, fontweight='bold')
    bx1.plot(time_sg2, sg2, color='black', label='signal')
    bx1.plot(time_envelope2, envelope2, color='red', label='envelope')
    bx1.set_xlim([10, 15])
    
    bx2.set_title('Run 2: Produced envelope spectrum', fontsize=20, fontweight='bold')
    bx2.plot(xf2, yf2, color='blue')
    bx2.set_xlabel('Frequency (Hz)', fontsize=15)
    bx2.set_ylabel('Power', fontsize=15)
    bx2.set_xlim([0, 10])
    
    bx3.set_title('Run 2: Mean PLV = ' + str("{:.2f}".format(plvMean2)), fontsize=20, fontweight='bold')
    bx3.plot(time2, plv2, 'ob')
    bx3.set_xlabel('Time (sec)', fontsize=15)
    bx3.set_ylabel('Speech Synchrony (PLV)', fontsize=15)
    bx3.set_ylim([0, 1])   
    
 
def Exclusion_Criteria(plv1, plv2, Test_Version):
    '''
    

    Parameters
    ----------
    plv1 : PLV of run1.
    plv2 : PLV of run2.
    Test_Version : Whether the test was Implicit (Fixed) or Explicit (Accelerated).

    Returns
    -------
    flagExclude : Returns 1 if any exclusion criteria is accomplished.
    plvs : Both PLVs in an array.

    '''
    plvs = np.zeros(2)
    plvs[0] = np.mean(plv1)
    plvs[1] = np.mean(plv2)
    
    flagExclude = 0

    for i in range(2):
        if plvs[i] > 0.9:
            print('Error! The plv of Run ' + str(i+1) + ' is too high.')
            flagExclude = 1
        if plvs[i] < 0.1:
            print('Error! The plv of Run ' + str(i+1) + ' is too low.')
            flagExclude = 1
        
   
    if Test_Version == 'ExpAcc':
        
        with open('LinearRegressions.pickle', 'rb') as handle:
            LR_ExpAcc = pickle.load(handle)
        
        plv2_Est = LR_ExpAcc['p1'] * plvs[0] + LR_ExpAcc['p2']
        plv2_min = plv2_Est - 1.96 * LR_ExpAcc['rmse']
        plv2_max = plv2_Est + 1.96 * LR_ExpAcc['rmse']
        
        if (plvs[1] < plv2_min or plvs[1] > plv2_max):
            print('Error! The plvs across Runs are not congruent!')
            flagExclude = 1
    
    elif Test_Version == 'ImpFix':
        
        with open('LinearRegressions.pickle', 'rb') as handle:
            LR_ImpFix = pickle.load(handle)
        
        plv2_Est = LR_ImpFix['p1'] * plvs[0] + LR_ImpFix['p2']
        plv2_min = plv2_Est - 1.96 * LR_ImpFix['rmse']
        plv2_max = plv2_Est + 1.96 * LR_ImpFix['rmse']
        
        if (plvs[1] < plv2_min or plvs[1] > plv2_max):
            print('Error! The plvs across Runs are not congruent!')
            flagExclude = 1
        
    else:
        print('Error! Please write correctly the Test Version.')
    
    return flagExclude, plvs


def Probability_High(plvs, Test_Version):
    '''
    

    Parameters
    ----------
    plvs : Array of PLVs (run1, run2).
    Test_Version : Whether the test was Implicit (Fixed) or Explicit (Accelerated).

    Returns
    -------
    None.

    '''
    if Test_Version == 'ExpAcc':
        
        with open('Gaussian_Mixture_Fits.pickle', 'rb') as handle:
            gm_ExpAcc = pickle.load(handle)
             
        speech_synch = np.mean(plvs)
        
        lows  = gm_ExpAcc['amp1'] * norm.pdf(speech_synch, gm_ExpAcc['mu1'], gm_ExpAcc['sgm1'])
        highs = gm_ExpAcc['amp2'] * norm.pdf(speech_synch, gm_ExpAcc['mu2'], gm_ExpAcc['sgm2'])
        
    elif Test_Version == 'ImpFix':
        
        with open('Gaussian_Mixture_Fits.pickle', 'rb') as handle:
            gm_ImpFix = pickle.load(handle)
         
        speech_synch = np.mean(plvs)
        
        lows  = gm_ImpFix['amp1'] * norm.pdf(speech_synch, gm_ImpFix['mu1'], gm_ImpFix['sgm1'])
        highs = gm_ImpFix['amp2'] * norm.pdf(speech_synch, gm_ImpFix['mu2'], gm_ImpFix['sgm2'])
        
    else:
        print('Error! Please write correctly the Test Version.')
    
    
    probHigh = np.true_divide(highs, (highs+lows))
    
    window = tkinter.Tk()
    window.title('Speech Synchrony Test Outcome')
    window.geometry('700x70')
    
    label = tkinter.Label(window, text = 'The participants degree of synchrony is: ' + str("{:.2f}".format(speech_synch)) + 
                          ' and its probability of being a HIGH synchronizer is ' + str("{:.2f}".format(probHigh))).grid(row=0)
    
    button_widget = tkinter.Button(window,text="OK", command=window.destroy).grid(row=1)
   
    tkinter.mainloop()
            