import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.fft import fft


# ---------------- part a----------------------

fs, audio = wav.read('data/kelas.wav')
audio = audio.astype(np.float32)

# Normalize audio
audio = audio / np.max(np.abs(audio))

# Remove silence
def remove_silence(audio, threshold=0.02):
    energy = np.abs(audio)
    indices = np.where(energy > threshold)[0]
    return audio[indices[0]:indices[-1]]

audio = remove_silence(audio)

# Parameters
frame_size = 300
overlap = 100
hop_size = frame_size - overlap
p = 20  # LPC order

# Framing
def enframe(signal, frame_size, hop_size):
    num_frames = 1 + (len(signal) - frame_size) // hop_size
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        start = i * hop_size
        frames[i, :] = signal[start:start+frame_size]
    return frames

# Apply Hamming window
def apply_hamming(frames):
    window = np.hamming(frames.shape[1])
    return frames * window

# LPC analysis
def lpc(signal, order):
    R = np.correlate(signal, signal, mode='full')
    R = R[len(R)//2:]
    r = R[:order+1]
    R_matrix = toeplitz(r[:-1])
    rhs = -r[1:]
    a = np.linalg.solve(R_matrix, rhs)
    a = np.insert(a, 0, 1)
    error = r[0] + np.sum(a[1:] * r[1:])
    return a, error

# Pitch estimation
def pitch_hps(frame, fs, fmin=50, fmax=500):
    spectrum = np.abs(fft(frame))
    spectrum = spectrum[:len(spectrum)//2]
    hps_spectrum = spectrum.copy()
    for h in range(2, 5):
        decimated = spectrum[::h]
        hps_spectrum[:len(decimated)] *= decimated
    peak_index = np.argmax(hps_spectrum)
    frequency = peak_index * fs / len(frame)
    if fmin <= frequency <= fmax:
        return frequency
    else:
        return 0.0

# Process
frames = enframe(audio, frame_size, hop_size)
frames_hamming = apply_hamming(frames)

num_frames = frames_hamming.shape[0]
A = np.zeros((p, num_frames))
G = np.zeros(num_frames)
F0 = np.zeros(num_frames)

for i in range(num_frames):
    frame = frames_hamming[i]
    a, error = lpc(frame, p)
    A[:, i] = a[1:]
    G[i] = np.sqrt(error)
    f0 = pitch_hps(frame, fs)
    F0[i] = f0

# Reconstruct signal
def synthesize(A, G, F0, frame_size, hop_size, fs):
    num_frames = A.shape[1]
    signal_length = frame_size + (num_frames - 1) * hop_size
    reconstructed = np.zeros(signal_length)

    for i in range(num_frames):
        a = np.concatenate(([1], -A[:, i]))
        gain = G[i]
        f0 = F0[i]
        
        if f0 == 0:
            # Unvoiced frame
            excitation = np.random.randn(frame_size)
        else:
            # Voiced frame
            pitch_period = int(fs / f0)
            excitation = np.zeros(frame_size)

            for n in range(0, frame_size, pitch_period):
                width = pitch_period // 3
                if width < 1:
                    width = 1
                end = min(n + width, frame_size)
                excitation[n:end] += np.hanning(end - n)

        excitation *= gain
        frame = signal.lfilter([1], a, excitation)
        start = i * hop_size
        reconstructed[start:start+frame_size] += frame
    
    return reconstructed

reconstructed_signal = synthesize(A, G, F0, frame_size, hop_size, fs)

# Save reconstructed signal
wav.write('data/kelas_reconstructed_overlap_hamming.wav', fs, reconstructed_signal.astype(np.float32))


#------------------Part (b)---------------------

# Parameters
hop_size_b = frame_size  # no overlap

# New frames
frames_b = enframe(audio, frame_size, hop_size_b)
num_frames_b = frames_b.shape[0]

A_b = np.zeros((p, num_frames_b))
G_b = np.zeros(num_frames_b)
F0_b = np.zeros(num_frames_b)

for i in range(num_frames_b):
    frame = frames_b[i]
    a, error = lpc(frame, p)
    A_b[:, i] = a[1:]
    G_b[i] = np.sqrt(error)
    f0 = pitch_hps(frame, fs)
    F0_b[i] = f0

# Reconstruct without Hamming and Overlap
reconstructed_signal_b = synthesize(A_b, G_b, F0_b, frame_size, hop_size_b, fs)

wav.write('data/kelas_reconstructed_nooverlap_nohamming.wav', fs, reconstructed_signal_b.astype(np.float32))
