# Voice/voice_register.py
import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import python_speech_features as psf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DB = os.path.join(BASE, "VoiceDB")
os.makedirs(VOICE_DB, exist_ok=True)

def _extract_embedding_from_array(audio, rate=16000):
    mfcc = psf.mfcc(audio, rate, numcep=13, nfilt=26)
    return np.mean(mfcc, axis=0).astype('float32')  # 13-d

def enroll_voice(username, duration=3, samplerate=16000):
    print(f"[VOICE] Recording {duration}s for {username} - speak now...")
    rec = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    audio = rec.reshape(-1).astype('float32')
    emb = _extract_embedding_from_array(audio, rate=samplerate)
    path = os.path.join(VOICE_DB, f"{username}.npy")
    np.save(path, emb)
    print(f"✅ Voice saved: {path}")
    return True
