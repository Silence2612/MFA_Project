import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import python_speech_features as psf

VOICE_DB = "VoiceDB"
os.makedirs(VOICE_DB, exist_ok=True)

def extract_voice_embedding(audio_path):
    rate, audio = wav.read(audio_path)
    mfcc = psf.mfcc(audio, rate, numcep=13, nfilt=26)
    embedding = np.mean(mfcc, axis=0)  # 13-d vector
    return embedding

def enroll_voice(username, duration=3):
    print(f"\n🎤 Recording voice for {username} ({duration} sec)...")
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1)
    sd.wait()

    audio_path = os.path.join(VOICE_DB, f"{username}.wav")
    wav.write(audio_path, 16000, audio)

    emb = extract_voice_embedding(audio_path)
    np.save(os.path.join(VOICE_DB, f"{username}.npy"), emb)

    print(f"✅ Voice enrollment complete for {username}\n")
