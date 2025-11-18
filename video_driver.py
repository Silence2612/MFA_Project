import cv2
import os

from Face.face_compare import compare_face
from Face.face_register import register_face

from Gesture.gesture_compare import compare_gesture
from Gesture.gesture_register import register_gesture

from Voice.voice_register import enroll_voice
from Voice.voice_compare import compare_voice

# ---------------------
# Helper
# ---------------------

def check_and_confirm(path, username, feature_name):
    """Check if a sample exists and ask user if they want to add another."""
    if os.path.exists(path):
        ans = input(f"\n⚠ {feature_name} for '{username}' already exists. Add another sample? (y/n): ").strip().lower()
        return ans == "y"
    return True  # No sample exists → OK to enroll


# ---------------------
# Main Driver
# ---------------------

def video_driver(frame_skip=5):

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not found")
        return

    print("\n🎥 Webcam started...")
    print("Keys:")
    print("  F = Enroll Face")
    print("  G = Enroll Gesture")
    print("  V = Enroll Voice")
    print("  U = Enroll FULL MFA (Face + Gesture + Voice)")
    print("  q = Quit\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --------------------------
        # RECOGNITION (same as old)
        # --------------------------
        if frame_count % frame_skip == 0:

            # FACE
            face_results = compare_face(frame)
            for res in face_results:
                x, y, w, h = res["box"]
                name = res["name"]
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # GESTURE
            gesture_res = compare_gesture(frame)
            if gesture_res:
                cv2.putText(frame, f"Gesture: {gesture_res}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("MFA Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        # --------------------------
        # ENROLLMENT LOGIC
        # --------------------------

        if key in [ord('f'), ord('F')]:
            username = input("\nEnter username for FACE enrollment: ").strip()
            path = f"FaceDB/{username}.npy"

            if check_and_confirm(path, username, "Face"):
                register_face(username, frame)
                print("✅ Face enrolled.\n")

        elif key in [ord('g'), ord('G')]:
            username = input("\nEnter username for GESTURE enrollment: ").strip()
            path = f"GestureDB/{username}.png"

            if check_and_confirm(path, username, "Gesture"):
                register_gesture(username, frame)
                print("✅ Gesture enrolled.\n")

        elif key in [ord('v'), ord('V')]:
            username = input("\nEnter username for VOICE enrollment: ").strip()
            path = f"VoiceDB/{username}.npy"

            if check_and_confirm(path, username, "Voice"):
                enroll_voice(username)
                print("✅ Voice enrolled.\n")

        elif key in [ord('u'), ord('U')]:
            username = input("\nEnter username for FULL MFA: ").strip()

            # FACE
            fpath = f"FaceDB/{username}.npy"
            if check_and_confirm(fpath, username, "Face"):
                register_face(username, frame)

            # GESTURE
            gpath = f"GestureDB/{username}.png"
            if check_and_confirm(gpath, username, "Gesture"):
                register_gesture(username, frame)

            # VOICE
            vpath = f"VoiceDB/{username}.npy"
            if check_and_confirm(vpath, username, "Voice"):
                enroll_voice(username)

            print("🔥 FULL MFA enrollment complete.\n")

        elif key == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
