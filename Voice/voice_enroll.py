import sounddevice as sd
import numpy as np
import faiss
import tensorflow as tf
import os
import pickle

# ====== LOAD YAMNET ======
yamnet_model = tf.lite.Interpreter(model_path="Models/yamnet.tflite")
yamnet_model.allocate_tensors()
input_details = yamnet_model.get_input_details()
output_details = yamnet_model.get_output_details()

SAVE_EMB = "VA/voice_embeddings.index"
SAVE_META = "VA/voice_labels.pkl"


def extract_embedding(audio):
    """Runs YAMNet TFLite model and returns the average embedding."""
    audio = audio.astype(np.float32)
    yamnet_model.set_tensor(input_details[0]['index'], audio)
    yamnet_model.invoke()
    embeddings = yamnet_model.get_tensor(output_details[0]['index'])
    return np.mean(embeddings, axis=0)


def record_audio(duration=3, sr=16000):
    print(f"🎙 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


def enroll_voice(user_id):
    audio = record_audio()
    emb = extract_embedding(audio).reshape(1, -1)

    # Create or load FAISS
    if os.path.exists(SAVE_EMB):
        index = faiss.read_index(SAVE_EMB)
        with open(SAVE_META, "rb") as f:
            labels = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(emb.shape[1])
        labels = []

    # Add embedding + label
    index.add(emb)
    labels.append(user_id)

    # Save
    faiss.write_index(index, SAVE_EMB)
    with open(SAVE_META, "wb") as f:
        pickle.dump(labels, f)

    print(f"✅ Voice enrolled for user: {user_id}")


if __name__ == "__main__":
    uid = input("Enter user ID for enrollment: ")
    enroll_voice(uid)
