import librosa
import soundfile as sf
from .recording_tool import record_audio


for i in range(6):
    record_audio(f'data/vowels/wovel{i+1}', 8000, 3)
    print(f'vowel {i+1} saved successfully!')

for i in range(6):
    signal, sr = librosa.load(f'data/vowels/vowel{i+1}.wav', sr=8000)

    # Remove silence
    trimmed_signal, index = librosa.effects.trim(signal, top_db=20)
    print(sr)

    sf.write(f'data/vowels/vowel{i+1}_trimmed.wav', trimmed_signal, sr)
