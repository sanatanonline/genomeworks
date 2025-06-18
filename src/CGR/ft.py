import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Sampling parameters
fs = 500  # Sampling frequency (Hz)
T = 1.0   # Duration (seconds)
t = np.linspace(0, T, fs, endpoint=False)  # Time vector

# Create a signal composed of multiple sine waves
freqs = [5, 50, 120]  # Frequencies in Hz
signal = np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)

# Compute the Fast Fourier Transform (FFT)
y_fft = fft(signal)
n = fs // 2  # Only half of the spectrum is needed
freq = np.fft.fftfreq(fs, d=1/fs)[:n]  # Frequency axis
amplitude = 2.0 / fs * np.abs(y_fft[:n])  # Normalize amplitude

# Plot the time-domain signal
plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Original Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Time-Domain Signal')
plt.legend()

# Plot the frequency-domain representation
plt.subplot(2, 1, 2)
plt.plot(freq, amplitude, 'r', label='FFT Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum')
plt.legend()
plt.tight_layout()
plt.show()