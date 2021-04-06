# -*- coding: utf-8 -*-
"""
@author: Jesús Salazar Araya, Angello Crawford Clark
"""
#Librerías utilizadas
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.fft import ifft
from scipy import signal

# Parámetros
tasa_muestreo = 1024
deltaT = 1

# Tamaño del arreglo de muestras
nPuntos = deltaT*tasa_muestreo

#Puntos del eje x, tiempo
puntos_tiempo = np.linspace(0, deltaT, nPuntos)

#A continuación, se definen los parámetros de las tres señales que constituyen
#la señal construida y se definen las frecuencias y magnitudes de las tres señales.

frec_1 = 150
magnitud_1 = 35

frec_2 = 50
magnitud_2 = 50

frec_3 = 200
magnitud_3 = 50

# Señales
señal_1 = magnitud_1*np.sin(2*np.pi*frec_1*puntos_tiempo) #Señal sinusoidal
señal_2 = magnitud_2*signal.square(2 *np.pi*frec_2*puntos_tiempo) #Señal cuadrada
señal_3 = magnitud_3*signal.sawtooth(2*np.pi*frec_3*puntos_tiempo) #Señal triangular


# Ruido para la señal
ruido = np.random.normal(0, 20, nPuntos)

señal_pura = señal_1 + señal_2 + señal_3 #Esta señal es la suma de las tres señales sin ruido
señal_ruidosa = señal_1 + señal_2 + señal_3 + ruido #Esta es la señal ruidosa que se pretende filtrar

#En este apartado se grafica la señal original junto con la señal ruidosa

#Procedimiento para graficar la señal original
fig, (ax1, ax2) = plt.subplots(1, 2, dpi=120, sharey= True)
ax1.plot(puntos_tiempo[0:50], señal_pura[0:50])
ax1.set_title('Señal original')
ax1.set_xlabel('Tiempo')
ax1.set_ylabel('Amplitud')

#Procedimiento para graficar la señal ruidosa
ax2.plot(puntos_tiempo[1:50], señal_ruidosa[1:50])
ax2.set_title('Señal ruidosa')
ax2.set_xlabel('Tiempo')

plt.show()

#==============================================

# Aplicación de la transformada de Fourier

#Los puntos de frecuencia representan el eje x en las gráficas de frecuencia
puntos_frecuencia = np.linspace (0.0, 512, int(nPuntos/2))

# Se aplica la transformada rapida a la señal

#La función fft() permite calcular la transformada de Fourier
#Esta pasa la señal del dominio del tiempo al dominio de la frecuencia con su respectiva amplitud.
transformada_señal = fft(señal_ruidosa)

#Se calculan las amplitudes que contienen valores positivos únicamente.
amplitudes = (2/nPuntos)*np.abs(transformada_señal[0:np.int(nPuntos/2)])

#Procedimiento para graficar la señal ruidosa en el dominio de la frecuencia
fig, ax = plt.subplots(dpi=120)
ax.plot(puntos_frecuencia, amplitudes)
ax.set_title('Señal ruidosa en el dominio de la frecuencia')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud')
ax.set_xticks(np.arange(0,501,50))
plt.show()

#==============================================

# Filtrado de la señal

umbral = 15 #El umbral es el máximo permitido de amplitud en la señal

#La siguiente función es la encargada de filtrar la señal del ruido, eliminando aquellos
#valores que estén por debajo del umbral establecido
def Filtrar_señal(ampl, trans,um):
    #Mediante el siguiente ciclo for se filtra el ruido de la señal
    for iCont in range(len(amplitudes)):
        if amplitudes[iCont] < umbral:
            transformada_señal[iCont] = 0.0 + 0.0j
            amplitudes[iCont] = 0.0
    return  transformada_señal, amplitudes

#Se hace el llamado a la función
transformada_señal, amplitudes =  Filtrar_señal(amplitudes, transformada_señal,umbral)

#Procedimiento para graficar la señal filtrada en el dominio de la frecuencia
fig, ax = plt.subplots(dpi=120)
ax.plot(puntos_frecuencia, amplitudes)
ax.set_title('Señal filtrada en el dominio de la frecuencia')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud')
ax.set_xticks(np.arange(0,501,50))
plt.show()
 
#=============================================

# Aplicación de la transformada inversa y
# comparación con la señal original

transformada_señal_filtrada = transformada_señal #La transformada de la señal filtrada va a ser la transformada de la señal con sin el ruido

#La función ifft() permite calcular la transformada inversa de Fourier
#Esta pasa la señal del dominio de la frecuencia al dominio del tiempo con su respectiva amplitud.
señal_reconstruida = ifft(transformada_señal_filtrada)


#Procedimiento para graficar la señal original en el dominio del tiempo
fig, (ax1, ax2) = plt.subplots(1, 2, dpi=120, sharey= True)
ax1.plot(puntos_tiempo[0:50], señal_pura[0:50])
ax1.set_title('Señal original')
ax1.set_xlabel('Tiempo')
ax1.set_ylabel('Amplitud')

#Procedimiento para graficar la señal reconstruida en el dominio del tiempo
ax2.plot(puntos_tiempo[1:50], señal_reconstruida[1:50])
ax2.set_title('Señal reconstruida')
ax2.set_xlabel('Tiempo')
