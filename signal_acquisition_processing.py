# %% LIBRERIAS Y FUNCIONES
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from itertools import chain
from numpy.fft import fft, ifft
import pywt
import pywt.data
from scipy import integrate, signal
import scipy.signal
import scipy.io
from scipy.signal import butter, lfilter,hilbert, chirp, find_peaks
import matplotlib.backends.backend_pdf
import math as m
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os
from matplotlib.animation import FuncAnimation
from PIL import Image


# %% Acquisition 
#-----------------------------------------
user = os.getlogin()
folder1 = 'Datos'
date_exp = '20240122'
serie = 'Serie 5'
num_imp = ['100', '200', '300', '400']
signal_name = 'Voltage Divider'

#-----------------------------------------

signal_2_channel = {
  "Voltage Divider": "Ch1",
  "Ferrite": "Ch2",
  "Bioinspired": "Ch3",
  "Vivaldi": "Ch4"
}

signal_2_color = {
  "Voltage Divider": "red",
  "Ferrite": "black",
  "Bioinspired": "green",
  "Vivaldi": "blue"
}

# |--------------- CH1: IMPULSE --------------- CH2: HFCT --------------- CH3: BIO --------------- CH4: VIVALDI---------------|

dirr_100 = r'C:\Users\\' + user + '\\OneDrive - Universidad Técnica Federico Santa María\Lightning Impulse UHF\\' + folder1 + '\\' + date_exp + '\\' + serie + '\\' + num_imp[0] + '\\' + signal_2_channel[signal_name]
dirr_200 = r'C:\Users\\' + user + '\\OneDrive - Universidad Técnica Federico Santa María\Lightning Impulse UHF\\' + folder1 + '\\' + date_exp + '\\' + serie + '\\' + num_imp[1] + '\\' + signal_2_channel[signal_name]
dirr_300 = r'C:\Users\\' + user + '\\OneDrive - Universidad Técnica Federico Santa María\Lightning Impulse UHF\\' + folder1 + '\\' + date_exp + '\\' + serie + '\\' + num_imp[2] + '\\' + signal_2_channel[signal_name]
dirr_400 = r'C:\Users\\' + user + '\\OneDrive - Universidad Técnica Federico Santa María\Lightning Impulse UHF\\' + folder1 + '\\' + date_exp + '\\' + serie + '\\' + num_imp[3] + '\\' + signal_2_channel[signal_name]


s_100 = pd.read_csv(dirr_100 + '.csv', header=None)
time = np.arange(len(s_100[0]))*(1/(5e9))*1e6
for i in range(int(len(s_100.columns)/2)):
    del s_100[2*i]
s_200 = pd.read_csv(dirr_200 + '.csv', header=None)
for i in range(int(len(s_200.columns)/2)):
    del s_200[2*i]
s_300 = pd.read_csv(dirr_300 + '.csv', header=None)
for i in range(int(len(s_300.columns)/2)):
    del s_300[2*i]
s_400 = pd.read_csv(dirr_400 + '.csv', header=None)      
for i in range(int(len(s_400.columns)/2)):
    del s_400[2*i]  
    
s_100 = s_100.reset_index(drop=True)
s_200 = s_200.reset_index(drop=True)
s_300 = s_300.reset_index(drop=True)
s_400 = s_400.reset_index(drop=True)

s_t = pd.concat([s_100, s_200, s_300, s_400], axis=1) 

s_t.columns = range(s_t.shape[1])

del s_100
del s_200
del s_300
del s_400

# %%                                                [trigger, width]
def signal_process_plot(i, plt_enable, calc_enable, peak_param, peak, tx, ty):    
  
        senal_t = s_t[i]  
        # senal_t = butter_bandpass_filter(senal_t, f1_uhf, f2_uhf, fs, order=5)        

        f , senal_f = signal.welch(senal_t, 5e9, nperseg = 1000, scaling = 'spectrum')
        f = f*1e-9   
        E = []
        dp_peaks = []
        max_peaks = []
        vpp = []
        
        if calc_enable:        
            E = time_power(senal_t)
            vp = max(senal_t)
            vpp = vp + abs(min(senal_t))
            # senal_t = senal_t/max(max(senal_t), abs(min(senal_t)))
            dp_peaks, _ = scipy.signal.find_peaks(senal_t, height=peak_param[0], distance = peak_param[1])
        
        if plt_enable:
        
            # pparam = dict(xlabel='Frequency (GHz)', ylabel='$V^{2}$')
            # fig, ax = plt.subplots(figsize = (6,3))
    
            # plt.plot(f, senal_f, linestyle = 'solid', label = signal_name, color = 'red', linewidth=1)           
              
            # plt.title(signal_name + ' Signal N° ' + str(i) + ' ' + serie)
            # ax.legend(title='Signal', prop={'size': fonttt})
            # ax.set_xlim([0,1])
            # # ax.set_ylim([0,50e-7])
            # ax.set(**pparam)
            
            pparam = dict(xlabel='Time ($\mu s$)', ylabel='$V$')
            fig, ax = plt.subplots(figsize = (6,3))
    
            plt.plot(time, senal_t, linestyle = 'solid', label = signal_name, color = 'red', linewidth=1)     
            plt.axhline(y = peak_param[0], color = 'black', label = 'Peak Trigger', linewidth = 0.8)
            plt.scatter(time[dp_peaks], senal_t[dp_peaks], marker='x', color = 'black', s = 15)
              
            plt.title(signal_name + ' Signal N° ' + str(i) + ' ' + serie)
            ax.legend(title='Signal', prop={'size': fonttt})
            ax.set_xlim(tx)
            ax.set_ylim(ty)
            ax.set(**pparam)
            
            # pparam = dict(xlabel='Time ($\mu s$)', ylabel='$V$')
            # fig, ax = plt.subplots(figsize = (6,3))
    
            # plt.plot(time[peak-2500:peak+2500], senal_t[peak-2500:peak+2500], linestyle = 'solid', label = signal_name, color = 'red', linewidth=1)           
              
            # plt.title(signal_name + ' Signal N° ' + str(i) + ' ' + serie)
            # ax.legend(title='Signal', prop={'size': fonttt})
            # # ax.set_xlim([0,50])
            # # ax.set_ylim([0,50e-7])
            # ax.set(**pparam)
        
        return E, dp_peaks, vp, vpp       
