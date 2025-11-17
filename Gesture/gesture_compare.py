# gesture_compare.py
import cv2
import numpy as np
from .gesture_register import get_hand_embedding   # ← FIXED

# Load saved embeddings
saved_gestures = np.load("gesture_embeddings.npy", allow_pickle=True)

def compare_gesture(frame, threshold=0.5):
    """
    Takes: OpenCV frame (BGR)
    Returns: gesture label or 'Unknown'
    """

    embedding = get_hand_embedding(frame)
    if embedding is None:
        return None  # no hand detected

    best_label = "Unknown"
    best_dist = float("inf")

    for item in saved_gestures:
        dist = np.linalg.norm(embedding - item["embedding"])
        if dist < best_dist:
            best_dist = dist
            best_label = item["label"]

    # Debug print (optional)
    # print(f"[GESTURE] {best_label} (dist={best_dist:.4f})")

    return best_label if best_dist < threshold else "Unknown"
