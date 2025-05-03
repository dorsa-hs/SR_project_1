import sounddevice as sd
from scipy.io.wavfile import write


def record_audio(filename, fs, duration):
    print("start recording")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(f"data/{filename}.wav", fs, recording)
    print("audio file saved successfuly!")
    
    
# record_audio("sentence", 8000, 20)