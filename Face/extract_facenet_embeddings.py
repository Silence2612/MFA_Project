import numpy as np
import os
import cv2
from keras_facenet import FaceNet

embedder = FaceNet()

faces_folder = os.path.join(os.path.dirname(__file__), 'faces')
face_data = []

for filename in os.listdir(faces_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(faces_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Skipping {filename}, could not load image.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        embedding = embedder.embeddings([image])[0]

        face_data.append({
            "name": os.path.splitext(filename)[0],
            "embedding": embedding
        })

# Save embeddings
np.save("face_embeddings.npy", face_data, allow_pickle=True)
print("✅ Saved embeddings to face_embeddings.npy")

# --- Load it later ---
loaded_data = np.load("face_embeddings.npy", allow_pickle=True)
print("🔁 Loaded", len(loaded_data), "embeddings.")
