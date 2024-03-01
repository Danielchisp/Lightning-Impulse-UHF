import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io
import pywt
import pywt.data
import os
import sys
from tqdm import tqdm

class Signal_Acquisition_Analyzer:

    @staticmethod
    def time_power(x):
        energia = 0
        for i in range(len(x)):
            energia = energia + x[i]**2     
        return energia

    def __init__(self, serie, signal_name):
        self.serie = serie
        self.signal_name = signal_name
        self.datos = self.load_data()

    def load_data(self):
        user = os.getlogin()
        folder = 'Datos'
        date_exp = '20240122'
        num_imp = ['100', '200', '300', '400']

        signal_channel = {
            "Voltage Divider": "Ch1",
            "Ferrite": "Ch2",
            "Bioinspired": "Ch3",
            "Vivaldi": "Ch4"
        }

        self.signal_color = {
            "Voltage Divider": "red",
            "Ferrite": "black",
            "Bioinspired": "green",
            "Vivaldi": "blue"
        }

        dirrs = [f'C:\\Users\\{user}\\OneDrive - Universidad Técnica Federico Santa María\\Lightning Impulse UHF\\{folder}\\{date_exp}\\{self.serie}\\{num}\\{signal_channel[self.signal_name]}' for num in num_imp]

        s_data = [pd.read_csv(f'{dirr}.csv', header=None, nrows=1250000) for dirr in dirrs]

        # Tomar la primera columna del primer DataFrame y asignarla a la variable 'time'
        self.time = s_data[0].iloc[:, 0].reset_index(drop=True)*(1e6)

        for i, s in enumerate(s_data):
            for j in range(int(len(s.columns) / 2)):
                del s[2 * j]
            s_data[i] = s.reset_index(drop=True)

        s_t = pd.concat(s_data, axis=1)
        s_t.columns = range(s_t.shape[1])
        self.s_t = s_t

        return s_t


    def metrics(self, peak_param, impulse_num):
        s_i = self.s_t[impulse_num]
        self.peak_param = peak_param

        E = Signal_Acquisition_Analyzer.time_power(s_i)        
        vp = max(s_i)
        vpp = vp + abs(min(s_i))        
        self.dp_peaks, _ = scipy.signal.find_peaks(peak_param[2]*s_i, height=peak_param[0], distance = peak_param[1])        

        metrics_out = {
        'E': E,
        'vp': vp,
        'vpp': vpp,
        'peaks': self.dp_peaks
        }
        return metrics_out


    def time_plot(self, impulse_num):
        
        fonttt = 8
        res = 150
        s_i = self.s_t[impulse_num]
        pparam = dict(xlabel='Time ($\mu s$)', ylabel='$V\, (Volts)$')
        # fig, ax = plt.subplots(figsize = (6,3))

        # plt.plot(self.time, s_i, linestyle = 'solid', label = self.signal_name, color = 'red', linewidth=1)     
        # # plt.axhline(y = self.peak_param[0], color = 'black', label = 'Peak Trigger', linewidth = 0.8)
        # # plt.scatter(self.time[self.dp_peaks], s_i[self.dp_peaks], marker='x', color = 'black', s = 30)
            
        # plt.title(self.signal_name + ' Signal N° ' + str(impulse_num) + ' ' + self.serie)
        # ax.legend(title='Signal', prop={'size': fonttt})
        # # ax.set_xlim(tx)
        # # ax.set_ylim(ty)
        # ax.set(**pparam)
        # plt.show()
        

        
    def wavelet_plot(self, impulse_num):
        
        res = 150
        s_i = self.s_t[impulse_num]
        pparam = dict(xlabel='Time ($\mu s$)', ylabel='$Frequency\, (GHz)$')
        coef, freqs = pywt.cwt(s_i, np.arange(1,res), 'morl',axis = 0, sampling_period=(1/(5e9)))
        freqs = freqs*1e-9
        fig, ax = plt.subplots(figsize = (6,3))
        wavelet = plt.contourf(self.time, freqs, abs(coef), cmap = 'plasma')
        ax.set_ylim(0, 1)
        ax.set(**pparam)
        
    def full_metrics(self, peak_param, imp_number):
        
        E = []
        dp_peaks = []
        vpp = []
        vp = []
        dp_count = []
        
        for i in tqdm(range(imp_number), desc="Metrics", unit="impulse"):
            out = self.metrics(peak_param,i)           
            
            E.append(out['E'])
            dp_peaks.append(list(out['peaks']))
            vp.append(out['vp'])
            vpp.append(out['vpp'])
            dp_count.append(len(out['peaks']))
            
            progress = (i + 1) / imp_number * 100
            sys.stdout.write(f'\rProgreso: {progress:.2f}%')
            sys.stdout.flush()
            
        self.metrics_plot(E, 'red', 1)
        self.metrics_plot(vpp, 'blue', 2)
        self.metrics_plot(vp, 'green', 3)
        self.metrics_plot(dp_count, 'black', 4)      
        
        self.metrics_export(E, dp_peaks, vp, vpp, dp_count)
        
        return E, dp_peaks, vp, vpp, dp_count
    
    def metrics_plot(self, y, color, num):
        dpi = 150
        num_dic = {
            1 : 'E',
            2 : '$V_{pp}$',
            3 : '$V_{p}$',
            4 : 'Number of PDs'}
        
        with plt.style.context(['science','std-colors']): 
            fig, ax = plt.subplots(dpi = dpi,figsize = (6,3))
            plt.plot(y, label = self.signal_name, color = color)
            pparam = dict(xlabel='Impulse number', ylabel=num_dic[num])
            plt.title(self.signal_name + ' - ' + self.serie)
            ax.legend(title='Signal', prop={'size': 8})
            # ax.set_xlim([0,2])
            # ax.set_ylim([0,0.05])
            ax.set(**pparam)
            # ax.set_ylim([0,0.5*1e-11])

            plt.show()
            
    def metrics_export(self, E, dp_peaks, vp, vpp, dp_count):
        metrics_df = pd.DataFrame({
            'E': E,
            'vpp': vpp,
            'vp': vp,
            'Number of DPs': dp_count
            })
        
        metrics_df.to_csv('Metrics ' + self.serie + ' ' + self.signal_name + '.csv', index=False)
        
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
        
        peaks_df = pd.DataFrame(result_matrix)
        peaks_df.to_csv('Peaks ' + self.serie + ' ' + self.signal_name + '.csv', index=False)
        
    def peaks_analyzer(self, file, plot_enable, energy_plot_enable, window_us):
        window_us = int(window_us*5000/2)
        peaks = pd.read_csv(file, header=0)
        E_pd = []
        vert_imp = []
        pd_count = 0
        
        for i in range(270,271):
            if (i==99 or i==199 or i==299):
                vert_imp.append(pd_count)
            for j in peaks.iloc[i]:                
                if j == -1:
                    break                
                if self.time[j] > 5:
                    pd_count = pd_count + 1
                
                    s_i = self.s_t[i]
                    s_i = s_i[j-window_us:j+window_us]
                    s_i = s_i.reset_index(drop=True)
                    
                    E_pd.append(Signal_Acquisition_Analyzer.time_power(s_i))            
                    
                    if plot_enable:
                        time_pd = self.time[j-window_us:j+window_us]
                        self.pd_time_plot(s_i, i, time_pd)
                        break
                        
            
        if energy_plot_enable:
            fonttt = 8
            pparam = dict(xlabel='Time ($\mu s$)', ylabel='$V\, (Volts)$')
            fig, ax = plt.subplots(figsize = (6,3))
    
            plt.plot(E_pd, linestyle = 'solid', label = self.signal_name, color = 'red', linewidth=1)     
            plt.axvline(x = vert_imp[0], color = 'black', label = '100', linewidth = 0.8)
            plt.axvline(x = vert_imp[1], color = 'blue', label = '200', linewidth = 0.8)
            plt.axvline(x = vert_imp[2], color = 'green', label = '300', linewidth = 0.8)
                
            plt.title(self.signal_name + ' Signal N° ' + ' ' + self.serie)
            ax.legend(title='Signal', prop={'size': fonttt})
            # ax.set_xlim(tx)
            # ax.set_ylim(ty)
            ax.set(**pparam)
            plt.show()       
                
        return peaks, E_pd
    
    def pd_time_plot(self, senal_peak, impulse_num, time_pd):
        
        fonttt = 8
        s_i = senal_peak
        pparam = dict(xlabel='Time ($\mu s$)', ylabel='$V\, (Volts)$')
        res = 300
        
        coef, freqs = pywt.cwt(s_i,np.arange(1,res),'morl',axis = 0,sampling_period=(1/(5e9)))
        fig, ax = plt.subplots(figsize = (6,3))
        wavelet = plt.contourf(time_pd, freqs[0:10], abs(coef[0:10]), cmap = 'plasma')
        # ax.set_ylim(1e9)
        # plt.style.use('default')
        # fig, ax = plt.subplots(figsize = (6,3))

        # plt.plot(time_pd, s_i, linestyle = 'solid', label = self.signal_name, color = 'red', linewidth=1)     
        # # plt.axhline(y = self.peak_param[0], color = 'black', label = 'Peak Trigger', linewidth = 0.8)
        # # plt.scatter(self.time[self.dp_peaks], s_i[self.dp_peaks], marker='x', color = 'black', s = 30)
            
        # plt.title(self.signal_name + ' Signal N° ' + str(impulse_num) + ' ' + self.serie)
        # ax.legend(title='Signal', prop={'size': fonttt})
        # # ax.set_xlim(tx)
        # # ax.set_ylim(ty)
        # ax.set(**pparam)
        # plt.show()
        
        
            
        
        
        
            
        
                
        
            
        
        
