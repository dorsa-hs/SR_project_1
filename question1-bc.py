import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

signal, sr = librosa.load("data/kelas.wav", sr=8000)

frame_length = int(0.025 * sr)  # 25 ms
hop_length = int(0.010 * sr)    # 10 ms

# Compute STE
ste = np.array([
    np.sum(np.square(signal[i:i + frame_length]))
    for i in range(0, len(signal) - frame_length, hop_length)
])

# Compute ZCR
zcr = librosa.feature.zero_crossing_rate(signal, frame_length=frame_length, hop_length=hop_length)[0]

# Normalize
ste_norm = (ste - np.min(ste)) / (np.max(ste) - np.min(ste))
zcr_norm = (zcr[:len(ste)] - np.min(zcr)) / (np.max(zcr) - np.min(zcr))

# Set thresholds
ste_silence_th = 0.01
ste_voiced_th = 0.25
zcr_unvoiced_th = 0.45

# Classify frames as S, U, or V
labels = []
for i in range(len(ste)):
    if ste_norm[i] < ste_silence_th:
        labels.append('S')  # Silence
    elif ste_norm[i] >= ste_voiced_th:
        labels.append('V')  # Vowel
    elif zcr_norm[i] > zcr_unvoiced_th:
        labels.append('U')  # Unvoiced consonant
    else:
        labels.append('S')  # uncertain cases as Silence

frames = range(len(ste))
t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

# Convert labels for plotting
label_map = {'S': 0, 'U': 1, 'V': 2}
label_numeric = [label_map[l] for l in labels]

# Plot
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
librosa.display.waveshow(signal, sr=sr)
plt.title('Audio Signal - Word "kelas"')

plt.subplot(3, 1, 2)
plt.plot(t, ste_norm, label='Normalized STE', color='green')
plt.plot(t, zcr_norm, label='Normalized ZCR', color='blue')
plt.axhline(ste_silence_th, color='green', linestyle='--', label='STE Silence Threshold')
plt.axhline(ste_voiced_th, color='green', linestyle=':', label='STE Voiced Threshold')
plt.axhline(zcr_unvoiced_th, color='blue', linestyle='--', label='ZCR Unvoiced Threshold')
plt.ylabel('Normalized Features')
plt.legend()

plt.subplot(3, 1, 3)
plt.step(t, label_numeric, where='mid', color='purple')
plt.yticks([0, 1, 2], ['Silence (S)', 'Unvoiced (U)', 'Vowel (V)'])
plt.xlabel('Time (s)')
plt.ylabel('Label')
plt.title('Frame-Level Classification')

plt.tight_layout()
plt.show()
