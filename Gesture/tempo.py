# extract_embeddings.py
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import sys
from gesture_register import get_detailed_embedding, hands

BASE = os.path.dirname(os.path.abspath(__file__))
GESTURE_FOLDER = os.path.join(BASE, "gestures")
OUTPUT_FILE = os.path.join(BASE, "gesture_embeddings.pkl")

def main():
    if not os.path.exists(GESTURE_FOLDER):
        print(f"❌ Folder '{GESTURE_FOLDER}' does not exist.")
        sys.exit()

    files = [f for f in os.listdir(GESTURE_FOLDER) if f.lower().endswith((".png",".jpg",".jpeg"))]
    gesture_db = {}

    for file in files:
        img_path = os.path.join(GESTURE_FOLDER, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Cannot read {file}")
            continue

        result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            print(f"❌ No hand detected in {file}")
            continue

        embedding = get_detailed_embedding(result.multi_hand_landmarks[0])
        gesture_db[os.path.splitext(file)[0]] = embedding
        print(f"✅ Processed: {file}")

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(gesture_db, f)

    print(f"\n🎉 Saved {len(gesture_db)} gestures to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
