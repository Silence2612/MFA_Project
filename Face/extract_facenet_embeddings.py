import numpy as np
import os
import cv2
from keras_facenet import FaceNet

# Initialize FaceNet embedder
embedder = FaceNet()

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_folder = os.path.join(os.path.dirname(__file__), 'faces')
face_data = []

for filename in os.listdir(faces_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(faces_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Skipping {filename}, could not load image.")
            continue

        # Convert to RGB (FaceNet expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = face_cascade.detectMultiScale(rgb_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) == 0:
            print(f"🚫 No face detected in {filename}, skipping.")
            continue

        # For each detected face (usually 1 per image)
        for (x, y, w, h) in faces:
            face_crop = rgb_image[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (160, 160))  # FaceNet default size

            # Get embedding
            embedding = embedder.embeddings([face_crop])[0]

            # Store name + embedding
            face_data.append({
                "name": os.path.splitext(filename)[0],
                "embedding": embedding
            })
            print(f"✅ Processed {filename}")

# Save embeddings
np.save("face_embeddings.npy", face_data, allow_pickle=True)
print("💾 Saved embeddings to face_embeddings.npy")

# --- Load it later ---
loaded_data = np.load("face_embeddings.npy", allow_pickle=True)
print("🔁 Loaded", len(loaded_data), "embeddings.")
