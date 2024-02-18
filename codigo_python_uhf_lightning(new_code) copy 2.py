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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def band_energy(v,nb):
    e_banda = []
    for k in range(len(nb)-1):
        init1 = int(nb[k])
        fin1 = int(nb[k+1])
        e_banda.append(np.mean(v[init1:fin1]))
    return e_banda     

def time_power(x):
    energia = 0
    for i in range(len(x)):
        energia = energia + x[i]**2     
    return energia

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

def rms(x):
    rms = np.sqrt(np.mean(x**2))
    return rms

def print_wavelet(x, y, init_f, final_f, init_t, final_t, titulo, etiq_x, etiq_y, res, ts):
    plt.rcParams["figure.autolayout"] = True
    t1 = np.where(x == find_nearest(x, init_t))[0][0]
    t2 = np.where(x == find_nearest(x, final_t))[0][0]
    # y = y/max(y)
    coef, freqs=pywt.cwt(y[t1:t2],np.arange(1,res),'morl',axis=0,sampling_period=ts)
    f11,f22 = np.where(freqs == find_nearest(freqs, final_f))[0][0],np.where(freqs == find_nearest(freqs, init_f))[0][0]
    plt.rcParams["figure.autolayout"] = True
    with plt.style.context(['science','ieee','std-colors']):
        fig, ax = plt.subplots(dpi=res)
        wavelet = plt.contourf(x[t1:t2], freqs[f11:f22]*1e-9, abs(coef[f11:f22]))                        
        plt.title(titulo)
        plt.xlabel(etiq_x)
        plt.ylabel(etiq_y)
        
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')
    
def set_size(w,h, ax=None):
#""" w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


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
        
def animate(i):
    ax.clear()
    # ax.axis('off')
    signal_process_plot(i, 1, 0, 0, 0, 0)
# %% Parámetros Procesamiento de señal - Gráficos   
# Filtro y frecuencia de sampleo

f1_uhf = 10e6
f2_uhf = 2499e6
fs = 5e9

# Trigger

level_h = 0.05
level_b = 0.1

# Intervalo de análisis de peaks

interval = 0.5*5000
match = 2

# Tiempos de peaks 

final_h = np.array([])
final_b = np.array([])
final_time_h = np.array([])
final_time_b = np.array([])

# Recta de separación

m_slope = 3.071
c_rect = -0.0371

# Configuración gráficos

fonttt = 8
matplotlib.rcParams.update({'font.size': fonttt})  
delay = 0.25 # Corección delay señales
offset = 2
delay_init = 0.45
pparam = dict(xlabel='Frequency (GHz)', ylabel='V')   
titulo = 'Lightning Impulse UHF - Sphere-Sphere Paper Dupont'  

# Filtro impulso LP

lp = 100e6

# %% Animación 
with plt.style.context(['science','std-colors']): 
    fig, ax = plt.subplots(dpi=300, figsize = (8,4))            
    ani = FuncAnimation(fig, animate, interval = 0.001, frames = 15)
    ani.save('prueba2' + '.gif', writer='pillow', fps=60)
    
def animation(file_name, num_signals):
    with plt.style.context(['science','std-colors']): 
        fig, ax = plt.subplots(dpi=300, figsize = (8,4))            
        ani = FuncAnimation(fig, animate, interval = 0.001, frames = num_signals)
        ani.save(file_name + '.gif', writer='pillow', fps=60)
        
# %% Signal test
peak_param = [0.002,0.05*5000]
signal_process_plot(350, 1, 1, peak_param, 0, [], [])

# %% Conversión MANUAL a EXCEL
num_dp_peaks = []
for i in dp_peaks:
    num_dp_peaks.append(len(i))
# Encontrar la dimensión máxima entre todas las sublistas
max_dimension = max(len(sublist) for sublist in dp_peaks)

# Crear una matriz cuadrada de NumPy con dimensiones máximas
result_matrix = np.full((len(dp_peaks), max_dimension), -1)

# Asignar valores a la matriz
for i, sublist in enumerate(dp_peaks):
    result_matrix[i, :len(sublist)] = sublist

# Imprimir la matriz resultante
print(result_matrix)  
# %%       
    
E = np.array([])
dp_peaks = []
vpp = np.array([])
vp = np.array([])

for i in range(400):
    aux_1, aux_2, aux_3, aux_4 = signal_process_plot(i, 0, 1, peak_param, 0, [0,50], [-0.05, 0.05])
    
    E = np.append(E,aux_1)
    dp_peaks.append(list(aux_2))
    vp = np.append(vp,aux_3)
    vpp = np.append(vpp,aux_4)
    
    print(str(round(100*i/400,2)) + '% completed')

# %%    

with plt.style.context(['science','std-colors']): 
    fig, ax = plt.subplots(dpi=300, figsize = (8,4))
    plt.plot(E, label = signal_name, color = signal_2_color[signal_name])
    pparam = dict(xlabel='Impulse number', ylabel='Number of PDs')
    plt.title(signal_name + ' Number of PDs Per Impulse - ' + serie)
    ax.legend(title='Signal', prop={'size': fonttt})
    # ax.set_xlim([0,2])
    # ax.set_ylim([0,0.05])
    ax.set(**pparam)
    # ax.set_ylim([0,0.5*1e-11])

    plt.show()
        
        # num_iteraciones = 100
        # ani = FuncAnimation(fig, update, frames=num_iteraciones, init_func=init, blit=True)
        
        # final_h = np.append(final_h,senal2[k_h])
        # final_b = np.append(final_b,senal3[k_b])

        # final_time_h = np.append(final_time_h,time_h)
        # final_time_b = np.append(final_time_b,time_b)
        
        # pparam = dict(xlabel='Time ($\mu$s)', ylabel='(pu)')

        # senal2_int = senal2.cumsum()
        # senal2_int = senal2_int/max(senal2_int)

        # senal3_int = senal3.cumsum() 
        # senal3_int = senal3_int/abs(min(senal3_int))

        # senal4_int = senal4.cumsum() 
        # senal4_int = senal4_int/abs(min(senal4_int))

        # with plt.style.context(['science','std-colors']): 
        #     fig, ax = plt.subplots(dpi=300, figsize = (8,4))
        #     plt.plot(time + 0.35, senal3_int, linestyle = 'solid', label = 'Cummulative Signal Antenna Bio DUT', color = 'green', linewidth = 1, alpha = 1)
            # plt.plot(time + 0.35, senal4_int, linestyle = 'solid', label = 'Cummulative Signal Antenna Bio SG', color = 'red', linewidth = 1, alpha = 1)
            # plt.plot(time, senal2_int, linestyle = 'solid', label = 'Cummulative Signal Antenna HFCT', color = 'red', linewidth = 1)
            # plt.plot(time, senal1, linestyle = 'solid', label = 'Impulse', linewidth = 1, color = 'blue')
            # ax.legend(title='Signal', prop={'size': fonttt})
            # ax.set(**pparam)
            # ax.set_xlim([5,15])
            # ax.set_ylim([0.4,0.55])
        
        # REUTILIZAR
        # plt.plot(time,senal4/max(senal4)+2.5,linestyle = 'solid', label = 'Bio Spark Gap', color = 'red')  
        
# %%
pparam = dict(xlabel='Time ($\mu$s)', ylabel='(pu)')

senal2_int = senal2.cumsum()
senal2_int = senal2_int/max(senal2_int)

senal3_int = senal3.cumsum() 
senal3_int = senal3_int/abs(min(senal3_int))

senal4_int = senal4.cumsum() 
senal4_int = senal4_int/abs(min(senal4_int))

with plt.style.context(['science','std-colors']): 
    fig, ax = plt.subplots(dpi=300, figsize = (8,4))
    plt.plot(time + 0.35, senal3_int, linestyle = 'solid', label = 'Cummulative Signal Antenna Bio DUT', color = 'green', linewidth = 1, alpha = 1)
    # plt.plot(time + 0.35, senal4_int, linestyle = 'solid', label = 'Cummulative Signal Antenna Bio SG', color = 'red', linewidth = 1, alpha = 1)
    # plt.plot(time, senal2_int, linestyle = 'solid', label = 'Cummulative Signal Antenna HFCT', color = 'red', linewidth = 1)
    plt.plot(time, senal1, linestyle = 'solid', label = 'Impulse', linewidth = 1, color = 'blue')
    ax.legend(title='Signal', prop={'size': fonttt})
    ax.set(**pparam)
    # ax.set_xlim([5,15])
    # ax.set_ylim([0.4,0.55])
# %% SEPARACION
main_dis = []
rev_dis_up = []
rev_dis_down = []

final_h = -1*final_h
for i in range(len(final_h)):
    # if final_b[i] > m_slope*final_h[i]+c_rect and final_h[i]<0.05:
    #     rev_dis_up.append(i)      
    # elif final_b[i] <= m_slope*final_h[i]+c_rect and final_h[i]<0.35:
    #     rev_dis_down.append(i)        
    # else:
    main_dis.append(i) 
        
print(str(len(main_dis)+len(rev_dis_down)+len(rev_dis_up)))

    
# %% Gráfico Correlación
pparam = dict(xlabel='V (pu) HFCT', ylabel='V (pu) BIO DUT') 
with plt.style.context(['science','std-colors']): 
    fig, ax = plt.subplots(dpi=300, figsize = (8,4))
    
    # plt.scatter(final_h[rev_dis_up],final_b[rev_dis_up], s = 2, label = 'Upper Reverse Discharge')
    # plt.scatter(final_h[rev_dis_down],final_b[rev_dis_down], s = 2, label = 'Lower Reverse Discharge', color = 'red')
    plt.scatter(final_h[main_dis],final_b[main_dis], s = 2, label = 'Main Discharge', color = 'green')
    
    titulo = 'Peak HFCT vs Peak BIO DUT - ' + type_s + ' Impulse' 
    plt.title(titulo)
    ax.legend(title='Signal', prop={'size': fonttt}, loc='upper left')
    ax.set(**pparam) 
    # plt.xticks(np.arange(0, .35, step=0.05))
    # plt.yticks(np.arange(0, .5, step=0.05))
    # ax.set_xlim([0,.35])
    # ax.set_ylim([0,.5])
    
    # plt.plot(peaks_HFCT)
    # plt.plot(peaks_BIO)
    # ax.set_xlim([180,240])
        # ax.set_ylim([0,1])
        # ax.set_ylim(bottom=0)

# print_wavelet(time, s3[1], 300, 1000, 1e-6, 2e-6, 'titulo', 'etiq_x', 'etiq_y', 300, 5e9)

# %% PHASE RESOLVED
pparam = dict(xlabel='Time ($\mu$s)', ylabel='V (pu)') 
with plt.style.context(['science','std-colors']): 
    fig, ax = plt.subplots(dpi=300, figsize = (8,4))
    plt.scatter(final_time_b[rev_dis_up],final_b[rev_dis_up] ,s = 2, label = 'Upper Reverse Discharge', color = 'blue') 
    plt.scatter(final_time_b[main_dis],final_b[main_dis] ,s = 2, label = 'Main Discharge', color = 'green')
    plt.scatter(final_time_b[rev_dis_down],final_b[rev_dis_down] ,s = 2, label = 'Lower Reverse Discharge', color = 'red')
       
    plt.plot(time,senal1,linestyle = 'solid', label = 'Impulse', linewidth = 1, color = 'black', alpha = 0.7)
    titulo = 'Separation BIO ' + type_s + ' Impulse' 
    plt.title(titulo)
    ax.legend(title='Signal', prop={'size': fonttt}, loc='best')
    ax.set(**pparam) 
    plt.xticks(np.arange(0, 250, step=50))
    plt.yticks(np.arange(0, 1, step=0.1))
        
