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

# Multiply STE and ZCR
combined = ste * zcr[:len(ste)]

frames = range(len(ste))
t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

# Plot
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
librosa.display.waveshow(signal, sr=sr)
plt.title('Audio Signal - Word "kelas"')

plt.subplot(4, 1, 2)
plt.plot(t, ste, label='Short-Term Energy (STE)', color='green')
plt.ylabel('STE')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t, zcr[:len(t)], label='Zero Crossing Rate (ZCR)', color='blue')
plt.ylabel('ZCR')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t, combined, label='STE * ZCR', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Combined Feature')
plt.legend()

plt.tight_layout()
plt.show()
