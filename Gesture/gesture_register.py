# Gesture/gesture_register.py
import os
import cv2
import pickle
import numpy as np
import mediapipe as mp

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GESTURE_DB = os.path.join(BASE, "GestureDB")
os.makedirs(GESTURE_DB, exist_ok=True)

GESTURE_DB_FILE = os.path.join(GESTURE_DB, "gesture_embeddings.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


def _extract_features(hand_landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    pts -= pts[0]
    scale = np.linalg.norm(pts[12]) + 1e-8
    pts /= scale

    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    d_flat = d[np.triu_indices(21, k=1)]

    def ang(a, b, c):
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    joints = [
        (0,1,2),(1,2,3),(2,3,4),
        (0,5,6),(5,6,7),(6,7,8),
        (0,9,10),(9,10,11),(10,11,12),
        (0,13,14),(13,14,15),(14,15,16),
        (0,17,18),(17,18,19),(18,19,20)
    ]

    angles = np.array([ang(pts[a], pts[b], pts[c]) for (a, b, c) in joints])

    return np.concatenate([pts.flatten(), d_flat, angles])


def register_gesture(username):
    """Capture hand gesture and save embedding under username."""
    cam = cv2.VideoCapture(0)
    print("[GESTURE REGISTER] Show gesture. Press SPACE to capture, ESC to cancel.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Camera error!")
            cam.release()
            return False

        cv2.imshow("Gesture Register", frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        key = cv2.waitKey(1) & 0xFF

        # CANCEL
        if key == 27:  # ESC
            print("❌ Cancelled.")
            cam.release()
            cv2.destroyAllWindows()
            return False

        # CAPTURE
        if key == 32:  # SPACE
            if not res.multi_hand_landmarks:
                print("❌ No hand detected, try again.")
                continue

            emb = _extract_features(res.multi_hand_landmarks[0])

            # Load old DB
            if os.path.exists(GESTURE_DB_FILE):
                with open(GESTURE_DB_FILE, "rb") as f:
                    db = pickle.load(f)
            else:
                db = {}

            db[username] = emb

            with open(GESTURE_DB_FILE, "wb") as f:
                pickle.dump(db, f)

            print(f"✅ Gesture saved for {username}")
            cam.release()
            cv2.destroyAllWindows()
            return True
