import sounddevice as sd
import numpy as np
import faiss
import tensorflow as tf
import pickle
import os

yamnet_model = tf.lite.Interpreter(model_path="Models/yamnet.tflite")
yamnet_model.allocate_tensors()
input_details = yamnet_model.get_input_details()
output_details = yamnet_model.get_output_details()

SAVE_EMB = "VA/voice_embeddings.index"
SAVE_META = "VA/voice_labels.pkl"


def extract_embedding(audio):
    audio = audio.astype(np.float32)
    yamnet_model.set_tensor(input_details[0]['index'], audio)
    yamnet_model.invoke()
    embeddings = yamnet_model.get_tensor(output_details[0]['index'])
    return np.mean(embeddings, axis=0)


def record_audio(duration=3, sr=16000):
    print(f"🎙 Speak for 3 seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


def compare_voice(thresh=1.10):
    if not os.path.exists(SAVE_EMB):
        return None, 999, False

    index = faiss.read_index(SAVE_EMB)
    with open(SAVE_META, "rb") as f:
        labels = pickle.load(f)

    audio = record_audio()
    emb = extract_embedding(audio).reshape(1, -1)

    D, I = index.search(emb, 1)
    dist = float(D[0][0])
    idx = int(I[0][0])
    user_id = labels[idx]

    if dist < thresh:
        return user_id, dist, True
    return None, dist, False


if __name__ == "__main__":
    uid, d, ok = compare_voice()
    print("Result:", uid, d, ok)
