# Face/face_register.py
import os
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DB = os.path.join(BASE, "FaceDB")
os.makedirs(FACE_DB, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def _face_from_frame(frame):
    # frame: BGR (OpenCV). MTCNN expects PIL or RGB ndarray.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # mtcnn returns cropped face PIL image (or torch Tensor) or None
    face = mtcnn(rgb)
    if face is None:
        return None
    # face is a tensor (3,H,W) normalized in [0,1]
    if isinstance(face, torch.Tensor):
        with torch.no_grad():
            face = face.to(device).unsqueeze(0)  # (1,3,160,160)
            emb = resnet(face).cpu().numpy()[0]
            return emb.astype('float32')
    else:
        return None


def register_face(username, from_frame=None, save_path=None):
    """
    If from_frame is provided (BGR), uses it. Otherwise opens webcam, captures one frame.
    Saves numpy embedding to FaceDB/<username>.npy
    Returns True on success.
    """
    if save_path is None:
        save_path = os.path.join(FACE_DB, f"{username}.npy")

    if from_frame is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam not available.")
            return False
        print("[FACE REGISTER] Press SPACE to capture, ESC to cancel.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from webcam.")
                cap.release()
                return False
            display = frame.copy()
            cv2.putText(display, f"Face enroll: {username} (SPACE=save, ESC=cancel)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Face Enroll", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cap.release()
                cv2.destroyWindow("Face Enroll")
                return False
            if key == 32:  # SPACE
                _, frame = cap.read()
                cap.release()
                cv2.destroyWindow("Face Enroll")
                break

    emb = _face_from_frame(frame)
    if emb is None:
        print("❌ No face detected. Try again with clearer framing / light.")
        return False

    np.save(save_path, emb)
    print(f"✅ Face embedding saved: {save_path}")
    return True
