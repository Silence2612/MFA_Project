# face_register.py
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

embedder = InceptionResnetV1(pretrained='vggface2').eval()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')


def register_faces(folder_path="faces", save_file="face_embeddings.npy"):
    face_data = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Skipping {filename}, could not load image.")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(rgb, 1.1, 5, minSize=(120, 120))
        if len(faces) == 0:
            print(f"🚫 No face detected in {filename}, skipping.")
            continue

        (x, y, w, h) = faces[0]
        crop = rgb[y:y+h, x:x+w]
        crop = cv2.resize(crop, (160, 160))

        # Preprocess tensor
        tensor = torch.from_numpy(crop).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            embedding = embedder(tensor).numpy()[0]

        face_data.append({
            "name": os.path.splitext(filename)[0],
            "embedding": embedding
        })

        print(f"✅ Registered: {filename}")

    np.save(save_file, face_data, allow_pickle=True)
    print(f"💾 Saved embeddings to {save_file}")

    return face_data
