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

import os
import pandas as pd
import numpy as np

class Signal_Acquisition_Analyzer:
    def __init__(self, serie, signal_name):
        """
        Constructor de la clase. Se llama automáticamente al crear una instancia.

        Args:
        - serie (str): Nombre de la serie.
        - signal_name (str): Nombre de la señal.
        """
        self.serie = serie
        self.signal_name = signal_name
        self.datos = self.cargar_datos()

    def cargar_datos(self):
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

        signal_color = {
            "Voltage Divider": "red",
            "Ferrite": "black",
            "Bioinspired": "green",
            "Vivaldi": "blue"
        }

        dirrs = [f'C:\\Users\\{user}\\OneDrive - Universidad Técnica Federico Santa María\\Lightning Impulse UHF\\{folder}\\{date_exp}\\{self.serie}\\{num}\\{signal_channel[self.signal_name]}' for num in num_imp]

        s_data = [pd.read_csv(f'{dirr}.csv', header=None) for dirr in dirrs]

        for i, s in enumerate(s_data):
            for j in range(int(len(s.columns) / 2)):
                del s[2 * j]
            s_data[i] = s.reset_index(drop=True)

        s_t = pd.concat(s_data, axis=1)
        s_t.columns = range(s_t.shape[1])

        return s_t

    def analizar_datos(self):
        """
        Función para analizar los datos cargados.

        Returns:
        - resultado: El resultado del análisis (puedes ajustarlo según tus necesidades).
        """
        # Lógica para analizar los datos
        # Por ejemplo, puedes realizar operaciones en los datos cargados:
        resultado = self.datos.describe()
        return resultado


    def analizar_datos(self):
        """
        Función para analizar los datos cargados.

        Returns:
        - resultado: El resultado del análisis (puedes ajustarlo según tus necesidades).
        """
        # Lógica para analizar los datos
        # Por ejemplo, puedes realizar operaciones en los datos cargados:
        resultado = self.datos.describe()
        return resultado