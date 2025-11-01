import os
import cv2
import numpy as np
from keras_facenet import FaceNet

# Initialize FaceNet
embedder = FaceNet()

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load saved embeddings
saved_faces = np.load("face_embeddings.npy", allow_pickle=True)

def recognize_face(image_path, threshold=0.8):
    """
    Detects face, extracts embedding, compares with saved embeddings,
    and returns the closest match name (if below threshold).
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face(s)
    faces = face_cascade.detectMultiScale(rgb_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        print("🚫 No face detected.")
        return "Unknown"

    # Use the first detected face
    (x, y, w, h) = faces[0]
    face_crop = rgb_image[y:y+h, x:x+w]
    face_crop = cv2.resize(face_crop, (160, 160))  # FaceNet expects 160×160

    # Generate embedding for cropped face
    input_embedding = embedder.embeddings([face_crop])[0]

    best_match_name = "Unknown"
    best_distance = float('inf')

    # Compare with saved embeddings
    for person in saved_faces:
        saved_embedding = person["embedding"]
        distance = np.linalg.norm(input_embedding - saved_embedding)

        if distance < best_distance:
            best_distance = distance
            best_match_name = person["name"]

    print(f"Closest match: {best_match_name} (distance: {best_distance:.4f})")

    # Apply threshold
    if best_distance > threshold:
        return "Unknown"
    else:
        return best_match_name


# --- Example usage ---
test_image = os.path.join(os.path.dirname(__file__), "faces", "fufu.jpg")
result = recognize_face(test_image)
print("🧾 Recognized as:", result)
