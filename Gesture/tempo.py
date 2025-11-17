import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
GESTURE_FOLDER = os.path.join(BASE, "gestures")
OUTPUT_FILE = "gesture_embeddings.pkl"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)


def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords)


def main():
    print("🔍 Checking gesture folder...")

    if not os.path.exists(GESTURE_FOLDER):
        print(f"❌ Folder '{GESTURE_FOLDER}' does not exist.")
        sys.exit()

    files = os.listdir(GESTURE_FOLDER)
    image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    print(f"📁 Total files in folder: {len(files)}")
    print(f"🖼️ Image files found: {len(image_files)}")

    if not image_files:
        print("❌ No gesture images found. Exiting.")
        sys.exit()

    gesture_db = {}

    for file in image_files:
        img_path = os.path.join(GESTURE_FOLDER, file)
        print(f"\n➡ Processing: {img_path}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Cannot read {file}")
            continue

        # Show image for debugging
        cv2.imshow("Reading Image", img)
        cv2.waitKey(300)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            print(f"❌ No hand detected in {file}")
            continue

        # Extract embedding
        embedding = extract_landmarks(result.multi_hand_landmarks[0])

        # Save under filename
        gesture_name = os.path.splitext(file)[0]
        gesture_db[gesture_name] = embedding

        print(f"✅ Hand detected and saved: {gesture_name}")

    cv2.destroyAllWindows()

    if not gesture_db:
        print("❌ No embeddings created. Stopping.")
        sys.exit()

    # Save embeddings file
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(gesture_db, f)

    print(f"\n🎉 DONE! Saved {len(gesture_db)} gestures to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
