# driver.py
import cv2
from face_compare import compare_face
from gesture_compare_live import compare_gesture  # <-- live gesture version

def recognize_from_webcam(frame_skip=5):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam not found")
        return

    frame_count = 0
    print("🎥 Webcam started... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:

            # ---------------------------
            # FACE RECOGNITION
            # ---------------------------
            face_results = compare_face(frame)

            for res in face_results:
                x, y, w, h = res["box"]
                name = res["name"]
                dist = res["distance"]

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{name}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)

                print(f"[FACE] {name} (dist={dist:.4f})")

            # ---------------------------
            # GESTURE RECOGNITION
            # ---------------------------
            gesture_result = compare_gesture(frame)  # returns a label or "Unknown"

            if gesture_result is not None:
                cv2.putText(frame, f"Gesture: {gesture_result}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 255), 2)

                print(f"[GESTURE] {gesture_result}")


        cv2.imshow("Face + Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


recognize_from_webcam()
