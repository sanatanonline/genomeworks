import numpy as np
import matplotlib.pyplot as plt
import pywt

# Example DNA Sequence (Mutations introduced)
dna_sequence = "ATGCGATCGTACGTACGATCGATCGTACTGACTG"

# Step 1: Convert DNA Sequence to Numerical Signal
mapping = {'A': 1, 'T': -1, 'G': 2, 'C': -2}
numerical_signal = np.array([mapping[base] for base in dna_sequence])

# Step 2: Apply Continuous Wavelet Transform (CWT)
scales = np.arange(1, 20)  # Scale range for wavelet analysis
coefficients, _ = pywt.cwt(numerical_signal, scales, 'mexh')  # 'mexh' = Mexican Hat Wavelet

# Step 3: Plot Results
plt.figure(figsize=(12, 6))

# Plot original DNA signal
plt.subplot(2, 1, 1)
plt.plot(numerical_signal, marker='o', linestyle='-', color='b', label="DNA Signal")
plt.title("Numerical Representation of DNA Sequence")
plt.ylabel("Encoded Base Value")
plt.xlabel("Position in Sequence")
plt.legend()

# Plot Wavelet Transform Heatmap
plt.subplot(2, 1, 2)
plt.imshow(abs(coefficients), aspect='auto', cmap='hot', extent=[0, len(dna_sequence), 1, 20])
plt.colorbar(label="Wavelet Coefficient Magnitude")
plt.title("Wavelet Transform (Mexican Hat) - Mutation Detection")
plt.ylabel("Scale")
plt.xlabel("Position in Sequence")

plt.tight_layout()
plt.show()
