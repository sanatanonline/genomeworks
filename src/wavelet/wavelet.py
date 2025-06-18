import numpy as np
import pywt
import matplotlib.pyplot as plt

# DNA sequence
dna_sequence = "ATCGATCGATCGATCG"

# Mapping nucleotides to binary values
mapping = {'A': '00', 'T': '01', 'C': '10', 'G': '11'}
binary_sequence = ''.join([mapping[nucleotide] for nucleotide in dna_sequence])

# Convert binary sequence to numerical signal
signal = np.array([int(bit) for bit in binary_sequence])

# Apply Discrete Wavelet Transform (DWT)
coeffs = pywt.dwt(signal, 'haar')

# Plot the original signal and its wavelet coefficients
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(signal)
plt.title('Original Signal')
plt.xlabel('Index')
plt.ylabel('Value')

plt.subplot(2, 1, 2)
plt.plot(coeffs[0])
plt.title('Wavelet Coefficients')
plt.xlabel('Scale')
plt.ylabel('Coefficient Value')

plt.tight_layout()
plt.show()
