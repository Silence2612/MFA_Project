import cv2
import numpy as np
import faiss
from keras_facenet import FaceNet

# Initialize FaceNet and face detector
embedder = FaceNet()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load saved embeddings
saved_faces = np.load("face_embeddings.npy", allow_pickle=True)
embeddings = np.array([p["embedding"] for p in saved_faces]).astype('float32')
names = [p["name"] for p in saved_faces]

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def recognize_face_frame(frame, threshold=0.8):
    """Detect faces, compute embeddings, and return recognized names."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    recognized_names = []
    for (x, y, w, h) in faces:
        face_crop = rgb_frame[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (160, 160))

        embedding = embedder.embeddings([face_crop])[0].astype('float32')
        D, I = index.search(np.expand_dims(embedding, axis=0), k=1)

        best_distance = D[0][0]
        best_match = names[I[0][0]]
        name = best_match if best_distance < threshold else "Unknown"

        recognized_names.append(name)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        print(f"Detected: {name} (distance: {best_distance:.4f})")

    return frame, recognized_names


def recognize_from_webcam(frame_skip=5):
    print("🚀 Starting webcam initialization...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return
    print("🎥 Webcam accessed successfully.")

    frame_count = 0
    print("🎥 Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame, recognized = recognize_face_frame(frame)
            if recognized:
                print(f"Frame {frame_count}: {recognized}")

        cv2.imshow("FAISS Face Recognition - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


# --- Run Webcam Recognition ---
recognize_from_webcam(frame_skip=5)
