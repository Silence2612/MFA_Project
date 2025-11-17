# gesture_compare.py
import numpy as np
from .gesture_register import get_detailed_embedding, hands
import pickle
import os
import cv2


# Load saved gestures
pkl_path = os.path.join(os.path.dirname(__file__), "gesture_embeddings.pkl")
with open(pkl_path, "rb") as f:
    saved_gestures = pickle.load(f)

def compare_gesture(frame, threshold=7.5):
    """Compare current hand to saved gestures."""
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.multi_hand_landmarks:
        print("🚫 No hand detected")
        return None

    embedding = get_detailed_embedding(result.multi_hand_landmarks[0])

    best_label = "Unknown"
    best_dist = float("inf")
    for label, emb in saved_gestures.items():
        dist = np.linalg.norm(embedding - emb)
        # DEBUG
        print(f"{label}: {dist:.2f}")
        if dist < best_dist:
            best_dist = dist
            best_label = label

    print(f"🎯 Best match: {best_label} ({best_dist:.2f})")
    return best_label if best_dist < threshold else "Unknown"
