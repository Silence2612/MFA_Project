# Face/face_compare.py
import os
import numpy as np
import cv2
import torch
import faiss
from facenet_pytorch import MTCNN, InceptionResnetV1

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DB = os.path.join(BASE, "FaceDB")
os.makedirs(FACE_DB, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def _face_emb_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)
    if face is None:
        return None
    with torch.no_grad():
        face = face.to(device).unsqueeze(0)
        emb = resnet(face).cpu().numpy()[0].astype('float32')
    return emb


def _load_db():
    names, embs = [], []
    for f in os.listdir(FACE_DB):
        if f.lower().endswith(".npy"):
            name = os.path.splitext(f)[0]
            path = os.path.join(FACE_DB, f)
            emb = np.load(path).astype('float32')
            names.append(name)
            embs.append(emb)
    if embs:
        embs = np.stack(embs, axis=0)
    else:
        embs = np.zeros((0, 512), dtype='float32')
    return names, embs


# build index on import
NAMES, EMBS = _load_db()
if EMBS.shape[0] > 0:
    index = faiss.IndexFlatL2(EMBS.shape[1])
    index.add(EMBS)
else:
    index = None


def compare_face(frame, threshold=1.0):
    """
    Input: OpenCV frame (BGR).
    Output: (label, distance)
    """
    emb = _face_emb_from_frame(frame)
    if emb is None:
        return None, 9999  # always return 2 values

    if index is None or index.ntotal == 0:
        return None, 9999

    D, I = index.search(emb.reshape(1, -1), k=1)
    dist = float(D[0][0])

    if dist < threshold:
        label = NAMES[I[0][0]]
    else:
        label = "Unknown"

    return label, dist
