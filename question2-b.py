import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal


plt.figure(figsize=(12, 6))

vowel_names = ['a', 'e', 'o', 'aa', 'ee', 'oo']

for i in range(6):

    filename = f'data/vowels/vowel{i+1}_trimmed.wav'

    signal, sr = librosa.load(filename, sr=8000)

    # FFT
    N = len(signal)
    fft_spectrum = np.abs(np.fft.fft(signal))[:N // 2]
    freq = np.fft.fftfreq(N, d=1/sr)[:N // 2]

    # formants
    peaks, _ = scipy.signal.find_peaks(fft_spectrum, height=np.max(fft_spectrum) * 0.1, distance=20)
    formant_freqs = freq[peaks]
    formant_amps = fft_spectrum[peaks]

    # fundamental frequency (F0)
    f0, voiced_flag, voiced_probs = librosa.pyin(signal, fmin=50, fmax=500, sr=sr)
    f0_mean = np.nanmean(f0)  # Average F0 ignoring NaN

    # Plot
    ax = plt.subplot(3, 2 , i+1)
    ax.set_title(f'wovel /{vowel_names[i]}/')
    plt.plot(freq, fft_spectrum, label='FFT Spectrum', color='navy')

    # Mark formants
    formants = []
    for i in range(min(3, len(formant_freqs))):  # Only first 3 formants
        x = formant_freqs[i]
        y = formant_amps[i]
        formants.append(x)
        plt.plot(x, y, 'ro')
        plt.text(x + 20, y, f'{int(x)}', color='red', ha='left', va='bottom', fontsize=9)
    formants_array = np.array(formants, dtype=np.float64)
    print(f'Formant Frequencies from LPC: {formants_array}')
    
    # Mark F0
    if not np.isnan(f0_mean):
        f0_amp = np.interp(f0_mean, freq, fft_spectrum)
        plt.axvline(f0_mean, color='green', linestyle='--', label=f'F0 ≈ {int(f0_mean)} Hz')
        plt.text(f0_mean + 20, np.max(fft_spectrum) * 0.8, f'{int(f0_mean)} Hz', color='green', ha='left', va='bottom', fontsize=9)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend(['FFT Spectrum'])
    plt.grid(True)
    plt.xlim([0, 800])
    plt.tight_layout()
    

plt.tight_layout()
plt.show()