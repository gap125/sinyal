# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 22:19:23 2023

@author: Ghazi
"""

import numpy as np
import matplotlib.pyplot as plt

print("Nama : Ghazi Amalul Putra")
print("NRP  : 5009211010")

# Sinyal Asli
fs = 2000  
t = np.arange(0, 1, 1/fs) 
frequency = 20 
signal = np.sin(2 * np.pi * frequency * t)

# Sinyal Noise
noise = np.random.normal(0, 0.8, len(t))
noisy_signal = signal + noise

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, signal)
plt.title("Sinyal Asli")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal)
plt.title("Sinyal Noise")
plt.grid()

# Buat Filter Noise
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

cutoff_frequency = 20 
filtered_signal = butter_lowpass_filter(noisy_signal, cutoff_frequency, fs)

plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal)
plt.title(f"Sinyal Hasil Filter (Low-Pass {cutoff_frequency} Hz)")
plt.grid()
plt.tight_layout()

# Lakukan FFT pada Sinyal di Atas
fft_signal = np.fft.fft(signal)
fft_noisy_signal = np.fft.fft(noisy_signal)
fft_filtered_signal = np.fft.fft(filtered_signal)

frequencies = np.fft.fftfreq(len(t), 1/fs)

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(frequencies, np.abs(fft_signal))
plt.title("FFT dari Sinyal Asli")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(frequencies, np.abs(fft_noisy_signal))
plt.title("FFT dari Sinyal Noise")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(frequencies, np.abs(fft_filtered_signal))
plt.title("FFT dari Sinyal Hasil Filter")
plt.grid()

plt.tight_layout()
plt.show()
