import cv2
import os
import numpy as np
import sounddevice as sd

from Face.face_register import register_face
from Face.face_compare import compare_face

from Gesture.gesture_register import register_gesture
from Gesture.gesture_compare import compare_gesture

from Voice.voice_register import enroll_voice
from Voice.voice_compare import compare_voice


FACE_DB = "FaceDB"
GESTURE_DB = "GestureDB"
VOICE_DB = "VoiceDB"


def ensure_db():
    os.makedirs(FACE_DB, exist_ok=True)
    os.makedirs(GESTURE_DB, exist_ok=True)
    os.makedirs(VOICE_DB, exist_ok=True)


def ask_overwrite(label):
    """Ask if user wants to overwrite or extend."""
    print(f"\n⚠️  Data for '{label}' already exists.")
    print("Choose:")
    print("1 - Overwrite")
    print("2 - Cancel")
    ch = input("Enter choice: ").strip()
    return ch == "1"


# -----------------------------
# ENROLLMENT MENU
# -----------------------------
def enroll_user(username):
    while True:
        print(f"""
======================================
🔵 ENROLLMENT MODE — USER: {username}
======================================
Controls:
  F - Enroll Face
  G - Enroll Gesture
  V - Enroll Voice
  Q - Back to Main Menu
  ESC - Exit Program
""")

        key = input("Select (F/G/V/Q/ESC): ").strip().upper()

        if key == "Q":
            return  # return to main menu

        if key == "ESC":
            print("👋 Exiting program...")
            exit()

        # FACE ENROLL
        if key == "F":
            path = os.path.join(FACE_DB, f"{username}.npy")
            if os.path.exists(path):
                if not ask_overwrite(username):
                    continue

            print("[ENROLL] FACE")
            success = register_face(username)
            print("Face enroll result:", success)

        # GESTURE ENROLL
        elif key == "G":
            path = os.path.join(GESTURE_DB, f"{username}.pkl")
            if os.path.exists(path):
                if not ask_overwrite(username):
                    continue

            print("[ENROLL] GESTURE")
            ok = register_gesture(username)
            print("Gesture enroll result:", ok)

        # VOICE ENROLL
        elif key == "V":
            path = os.path.join(VOICE_DB, f"{username}.npy")
            if os.path.exists(path):
                if not ask_overwrite(username):
                    continue

            print("[ENROLL] VOICE")
            enroll_voice(username)
            print("Voice enroll complete.")

        else:
            print("❌ Invalid option! Use F/G/V/Q/ESC.")


# -----------------------------
# VERIFICATION MENU
# -----------------------------
def verify_menu(username):
    while True:
        print(f"""
======================================
🟣 VERIFICATION — USER: {username}
======================================
Choose what you want to verify:
  F - Verify Face
  G - Verify Gesture
  V - Verify Voice
  A - Verify ALL (Face + Gesture + Voice)
  Q - Back to Main Menu
  ESC - Exit Program
""")

        key = input("Select (F/G/V/A/Q/ESC): ").strip().upper()

        if key == "Q":
            return  # back to main menu

        if key == "ESC":
            print("👋 Exiting program...")
            exit()

        # Use camera only for F / G / A
        if key in ["F", "G", "A"]:
            cap = cv2.VideoCapture(0)
            print("🎥 Show your face/gesture to camera...")
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print("❌ Camera error.")
                continue

        # FACE ONLY
        if key == "F":
            face_label, face_dist = compare_face(frame)
            print(f"[DEBUG] Face: {face_label}  dist={face_dist}")
            print("Match:", face_label == username)

        # GESTURE ONLY
        elif key == "G":
            gesture_label, gesture_dist = compare_gesture(frame)
            print(f"[DEBUG] Gesture: {gesture_label}  dist={gesture_dist}")
            print("Match:", gesture_label == username)

        # VOICE ONLY
        elif key == "V":
            print("\n🎤 Speak for 2s...")
            voice_label, voice_dist = compare_voice(duration=2)
            print(f"[DEBUG] Voice: {voice_label} dist={voice_dist}")
            print("Match:", voice_label == username)

        # ENTIRE MFA
        elif key == "A":
            # FACE
            face_label, face_dist = compare_face(frame)
            print(f"[DEBUG] Face: {face_label} dist={face_dist}")

            # GESTURE
            gesture_label, gesture_dist = compare_gesture(frame)
            print(f"[DEBUG] Gesture: {gesture_label} dist={gesture_dist}")

            # VOICE
            print("\n🎤 Speak for 2s...")
            voice_label, voice_dist = compare_voice(duration=2)
            print(f"[DEBUG] Voice: {voice_label} dist={voice_dist}")

            ok_face = (face_label == username)
            ok_gesture = (gesture_label == username)
            ok_voice = (voice_label == username)

            print("\n==========================")
            print("🔎 RESULTS")
            print("==========================")
            print("Face Match:   ", ok_face)
            print("Gesture Match:", ok_gesture)
            print("Voice Match:  ", ok_voice)

            if ok_face and ok_gesture and ok_voice:
                print("\n✅ FULL VERIFICATION SUCCESS\n")
            else:
                print("\n❌ VERIFICATION FAILED\n")

        else:
            print("❌ Invalid option!")


# -----------------------------
# MAIN MENU LOOP
# -----------------------------
def video_driver():
    ensure_db()

    while True:
        print("""
===========================
  Multi-Factor System
===========================
1 - Enroll User
2 - Verify User
ESC - Quit Program
""")

        mode = input("Select (1/2/ESC): ").strip().upper()

        if mode == "ESC":
            print("👋 Exiting program...")
            break

        if mode == "1":
            username = input("Enter username to enroll: ").strip()
            enroll_user(username)

        elif mode == "2":
            username = input("Enter username to verify: ").strip()
            verify_menu(username)

        else:
            print("❌ Invalid choice! Use 1/2/ESC.")


if __name__ == "__main__":
    video_driver()
