# gesture_register.py
import os
import cv2
import numpy as np
import mediapipe as mp
import pickle

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def crop_hand(frame, hand_landmarks):
    """Crop hand bounding box from frame with padding."""
    h, w, _ = frame.shape
    x = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y = [int(lm.y * h) for lm in hand_landmarks.landmark]
    x_min, x_max = max(min(x) - 10, 0), min(max(x) + 10, w)
    y_min, y_max = max(min(y) - 10, 0), min(max(y) + 10, h)
    return frame[y_min:y_max, x_min:x_max]

def get_detailed_embedding(hand_landmarks):
    """Return detailed embedding: landmarks + distances + angles."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    pts -= pts[0]  # translate wrist to origin
    scale = np.linalg.norm(pts[12])
    if scale > 1e-6:
        pts /= scale

    # Pairwise distances
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    dists_flat = dists[np.triu_indices(pts.shape[0], k=1)]

    # Finger angles
    def angle(a, b, c):
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    finger_joints = [
        (0,1,2), (1,2,3), (2,3,4),
        (0,5,6), (5,6,7), (6,7,8),
        (0,9,10),(9,10,11),(10,11,12),
        (0,13,14),(13,14,15),(14,15,16),
        (0,17,18),(17,18,19),(18,19,20)
    ]
    angles = [angle(pts[a], pts[b], pts[c]) for a,b,c in finger_joints]

    return np.concatenate([pts.flatten(), dists_flat, angles])

def register_gestures(folder_path="gestures", save_file="gesture_embeddings.pkl"):
    gesture_db = {}
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' does not exist.")
        return gesture_db

    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for file in files:
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Cannot read {file}")
            continue

        result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            print(f"❌ No hand detected in {file}")
            continue

        embedding = get_detailed_embedding(result.multi_hand_landmarks[0])
        label = os.path.splitext(file)[0]
        gesture_db[label] = embedding
        print(f"✅ Registered gesture: {label}")

    # Save embeddings
    with open(save_file, "wb") as f:
        pickle.dump(gesture_db, f)
    print(f"💾 Saved {len(gesture_db)} gestures to {save_file}")

    return gesture_db

__all__ = ["get_detailed_embedding", "register_gestures"]
