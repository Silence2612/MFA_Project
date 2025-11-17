# face_compare.py
import cv2
import numpy as np
import faiss
import torch
from facenet_pytorch import InceptionResnetV1

# Load FaceNet (pretrained on VGGFace2)
embedder = InceptionResnetV1(pretrained='vggface2').eval()

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')

# Load saved embeddings
saved_faces = np.load("face_embeddings.npy", allow_pickle=True)
embeddings = np.array([p["embedding"] for p in saved_faces]).astype("float32")
names = [p["name"] for p in saved_faces]

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def compare_face(frame, threshold=0.9):
    """
    Input: full frame (BGR)
    Output: list of {name, distance, box}
    """
    results = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = face_cascade.detectMultiScale(rgb, 1.1, 5, minSize=(120, 120))

    for (x, y, w, h) in faces:
        crop = rgb[y:y+h, x:x+w]
        crop = cv2.resize(crop, (160, 160))

        # Preprocess for facenet-pytorch
        tensor = torch.from_numpy(crop).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1,3,160,160)

        with torch.no_grad():
            emb = embedder(tensor).numpy()[0].astype("float32")

        # FAISS search
        D, I = index.search(emb.reshape(1, -1), 1)

        dist = D[0][0]
        name = names[I[0][0]] if dist < threshold else "Unknown"

        results.append({
            "name": name,
            "distance": float(dist),
            "box": (x, y, w, h)
        })

    return results
