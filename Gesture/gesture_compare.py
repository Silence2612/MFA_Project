import os
import cv2
import numpy as np
import mediapipe as mp
import pickle

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GESTURE_DB_FILE = os.path.join(BASE, "GestureDB", "gesture_embeddings.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


def _extract_features(hand_landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    # Normalize to wrist
    pts -= pts[0]

    # Scale to middle finger length
    scale = np.linalg.norm(pts[12]) + 1e-8
    pts /= scale

    # Pairwise distances
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    d_flat = d[np.triu_indices(21, k=1)]

    # Angles between joints
    def ang(a, b, c):
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    joints = [
        (0,1,2),(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),
        (0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),
        (14,15,16),(0,17,18),(17,18,19),(18,19,20)
    ]
    angles = np.array([ang(pts[a], pts[b], pts[c]) for a, b, c in joints])

    return np.concatenate([pts.flatten(), d_flat, angles])


def compare_gesture(frame, threshold=10.0):
    """
    ALWAYS returns exactly:
        (label, distance)

    If no gesture or no DB:
        ("Unknown", 9999)
    """

    # No database available
    if not os.path.exists(GESTURE_DB_FILE):
        return "Unknown", 9999

    # Load gesture embeddings
    with open(GESTURE_DB_FILE, "rb") as f:
        db = pickle.load(f)

    # Run mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # No hand detected
    if not result.multi_hand_landmarks:
        return "Unknown", 9999

    # Extract embedding
    emb = _extract_features(result.multi_hand_landmarks[0])

    best_label = "Unknown"
    best_dist = float("inf")

    # Compare with DB
    for label, e in db.items():
        d = np.linalg.norm(emb - e)
        if d < best_dist:
            best_dist = d
            best_label = label

    # Apply threshold
    if best_dist >= threshold:
        best_label = "Unknown"

    return best_label, float(best_dist)
