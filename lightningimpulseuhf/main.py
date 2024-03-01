# %% Libraries
import pandas as pd
import numpy as np
import time
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
# %% Data Acquisition
from Signal_Acquisition_Analyzer import Signal_Acquisition_Analyzer

senal = Signal_Acquisition_Analyzer('Serie 6', 'Bioinspired')

# %% Parameters Adjustment
nro_señal = 320
peak_param = [0.0015,5000*0.1,1]

senal.metrics(peak_param,nro_señal)
senal.time_plot(nro_señal)

# %% Full Metrics
E, dp_peaks, vp, vpp, dp_count = senal.full_metrics(peak_param, 400)

# %% Peaks Analyzer
peaks_dirr = r'C:\Users\DanielFigueroa\OneDrive - Universidad Técnica Federico Santa María\Lightning Impulse UHF\scripts\lightningimpulseuhf\lightningimpulseuhf\Serie6_metrics/Peaks Serie 6 Bioinspired.csv'
peaks, e_pd = senal.peaks_analyzer(peaks_dirr,1,0,30)#.csv peaks, plot_enable, energy_plot_enable window us
# %%
nro_señal = 280
senal.wavelet_plot(nro_señal)