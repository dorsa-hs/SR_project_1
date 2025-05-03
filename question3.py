import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os
# from .recording_tool import record_audio

# # record the sentence
# record_audio("sentence", 8000, 20)

sample_rate, signal = wav.read('data/sentence.wav')

# Normalize
if signal.dtype == np.int16:
    signal = signal / 32768.0

# Frame parameters
frame_size = 0.03  # 30 ms
frame_shift = 0.01  # 10 ms

frame_len = int(frame_size * sample_rate)
frame_step = int(frame_shift * sample_rate)
signal_length = len(signal)
num_frames = int(np.ceil((signal_length - frame_len) / frame_step)) + 1

pad_signal_length = num_frames * frame_step + frame_len
pad_signal = np.append(signal, np.zeros(pad_signal_length - signal_length))

indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
          np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]

# Feature extraction
energy = np.sum(frames ** 2, axis=1)
zcr = np.sum(np.abs(np.diff(np.sign(frames))), axis=1) / (2 * frames.shape[1])

# Thresholds (can be tuned)
energy_threshold = 0.002 * np.max(energy)
zcr_threshold = 0.075 * np.max(zcr)

# Detect speech regions
is_speech = (energy > energy_threshold) | (zcr > zcr_threshold)
speech_regions = []
start = None

for i, val in enumerate(is_speech):
    if val and start is None:
        start = i
    elif not val and start is not None:
        if i - start >= 10:  # min_speech_frames
            speech_regions.append((start, i))
        start = None

if start is not None:
    if len(is_speech) - start >= 10:
        speech_regions.append((start, len(is_speech)))

print(f"Detected {len(speech_regions)} speech segments.")

# Plot
time = np.arange(len(signal)) / sample_rate
frame_time = np.arange(len(energy)) * frame_shift

plt.figure(figsize=(12, 8))

plt.subplot(3,1,1)
plt.plot(time, signal)
plt.title('Original Signal')
plt.xlabel('Time [s]')

plt.subplot(3,1,2)
plt.plot(frame_time, energy)
plt.title('Short-Time Energy')
plt.xlabel('Time [s]')

plt.subplot(3,1,3)
plt.plot(frame_time, zcr)
plt.title('Zero Crossing Rate')
plt.xlabel('Time [s]')

plt.tight_layout()
plt.show()

# Save each word separately
output_folder = 'data/words_output'
os.makedirs(output_folder, exist_ok=True)

for idx, (start, end) in enumerate(speech_regions):
    start_sample = start * frame_step
    end_sample = end * frame_step + frame_len
    word_signal = signal[start_sample:end_sample]
    
    output_path = os.path.join(output_folder, f'word_{idx+1}.wav')
    wav.write(output_path, sample_rate, (word_signal * 32767).astype(np.int16))

print(f"Saved {len(speech_regions)} words separately into '{output_folder}' folder.")
