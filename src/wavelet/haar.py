import math
import matplotlib.pyplot as plt


def haar_wavelet_transform(signal):
    N = len(signal)
    output = signal.copy()

    h = 1 / math.sqrt(2)
    step = 2

    while step <= N:
        temp = output.copy()
        for i in range(0, N, step):
            output[i // 2] = h * (temp[i] + temp[i + 1])
            output[(i // 2) + (N // step)] = h * (temp[i] - temp[i + 1])
        step *= 2

    return output


def create_signal(frequencies, amplitudes, sample_rate, duration):
    t = [i / sample_rate for i in range(int(sample_rate * duration))]
    signal = [0] * len(t)
    for i in range(len(frequencies)):
        freq = frequencies[i]
        amp = amplitudes[i]
        for j in range(len(t)):
            signal[j] += amp * math.sin(2 * math.pi * freq * t[j])
    return t, signal


# Parameters for the sample signal
sampling_rate = 16  # Keep it low for simplicity
duration = 1.0  # Duration in seconds
frequencies = [3, 5]  # Frequencies of the sine waves
amplitudes = [0.5, 0.3]  # Amplitudes of the sine waves

# Create a sample signal
t, signal = create_signal(frequencies, amplitudes, sampling_rate, duration)

# Compute the Haar Wavelet Transform of the signal
wavelet_transform = haar_wavelet_transform(signal)

# Plot the original signal and its Haar Wavelet Transform
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(t, wavelet_transform)
plt.title('Haar Wavelet Transform')
plt.xlabel('Time [s]')
plt.ylabel('Transform Value')

plt.tight_layout()
plt.show()
