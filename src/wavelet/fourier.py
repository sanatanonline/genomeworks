import numpy as np
import matplotlib.pyplot as plt

# Create a sample signal: a sum of two sine waves
sampling_rate = 1000  # Samples per second
T = 1.0  # Total time in seconds
t = np.linspace(0, T, int(T * sampling_rate), endpoint=False)
f1 = 50  # Frequency of the first sine wave
f2 = 120  # Frequency of the second sine wave
signal = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)

# Compute the Discrete Fourier Transform (DFT)
N = len(signal)
dft_result = np.zeros(N, complex)
for k in range(N):
    for n in range(N):
        dft_result[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)

# Compute the frequencies corresponding to the DFT results
frequencies = np.fft.fftfreq(N, 1/sampling_rate)

# Plot the signal and its Fourier Transform
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(frequencies[:N//2], np.abs(dft_result)[:N//2])
plt.title('Discrete Fourier Transform')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.xlim(0, 200)

plt.tight_layout()
plt.show()
