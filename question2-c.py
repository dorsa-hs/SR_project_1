import matplotlib.pyplot as plt
import numpy as np
import librosa
import scipy.signal

plt.figure(figsize=(10, 6))
vowel_names = ['a', 'e', 'o', 'aa', 'ee', 'oo']

for i in range(6):

    filename = f'data/vowels/vowel{i+1}_trimmed.wav'

    signal_trimmed, sr = librosa.load(filename, sr=8000)

    # LPC
    order = 16
    a = librosa.lpc(y=signal_trimmed, order=order)
    w, h = scipy.signal.freqz(1, a, worN=512, fs=sr)

    # plot
    ax = plt.subplot(3, 2, i+1)
    ax.set_title(f'LPC Spectrum of Vowel /{vowel_names[i]}/')
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()

    # formants
    formant_peaks, _ = scipy.signal.find_peaks(np.abs(h), distance=20)
    formant_freqs_lpc = w[formant_peaks[:3]]

    print(f"Formant Frequencies from LPC: {formant_freqs_lpc}")
    
    
plt.tight_layout()
plt.show()