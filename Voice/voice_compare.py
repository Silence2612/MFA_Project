import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import python_speech_features as psf
from scipy.spatial.distance import cosine

VOICE_DB = "VoiceDB"

def extract_voice_embedding_from_audio(audio):
    mfcc = psf.mfcc(audio, 16000, numcep=13, nfilt=26)
    return np.mean(mfcc, axis=0)

def compare_voice(duration=2, threshold=0.25):
    print("\n🎤 Listening for verification...")
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1)
    sd.wait()

    audio = audio.reshape(-1)
    emb_live = extract_voice_embedding_from_audio(audio)

    best_user = "Unknown"
    best_score = 999

    for f in os.listdir(VOICE_DB):
        if f.endswith(".npy"):
            user = f[:-4]
            stored_emb = np.load(os.path.join(VOICE_DB, f))

            dist = cosine(emb_live, stored_emb)
            if dist < best_score:
                best_score = dist
                best_user = user

    return best_user, best_score
