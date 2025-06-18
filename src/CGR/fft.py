import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def read_fasta(file_path):
    sequence = ""
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('>'):
                sequence += line.strip()
    return sequence

# Function to convert DNA sequence to numerical values
def dna_to_numeric(seq):
    mapping = {'A': 1, 'T': -1, 'C': 0.5, 'G': -0.5}
    return np.array([mapping[nuc] for nuc in seq if nuc in mapping])

# Example DNA sequence
# dna_sequence = "ATGCGTACGTAGCTAGCTAGCTAGT"
fasta_file = 'US1.fasta'
dna_sequence = read_fasta(fasta_file)

# Convert DNA to numerical sequence
numerical_seq = dna_to_numeric(dna_sequence)

# Compute FFT
y_fft = fft(numerical_seq)
n = len(numerical_seq) // 2  # Only half of the spectrum is needed
freq = np.fft.fftfreq(len(numerical_seq), d=1)[:n]  # Frequency axis
amplitude = np.abs(y_fft[:n])  # Magnitude spectrum

# Plot the DNA numerical sequence
plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.plot(numerical_seq, marker='o', linestyle='-', label='Numerical DNA Sequence')
plt.xlabel('Position')
plt.ylabel('Numerical Value')
plt.title('Numerical Representation of DNA')
plt.legend()

# Plot the frequency-domain representation
plt.subplot(2, 1, 2)
plt.plot(freq, amplitude, 'r', label='FFT Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum of DNA Sequence')
plt.legend()
plt.tight_layout()
plt.show()