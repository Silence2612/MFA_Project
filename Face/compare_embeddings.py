import os
import cv2
import numpy as np
from keras_facenet import FaceNet

# Initialize FaceNet
embedder = FaceNet()

# Load saved embeddings from .npy
saved_faces = np.load("face_embeddings.npy", allow_pickle=True)

def recognize_face(image_path, threshold=0.8):
    """
    Compares input image embedding with saved embeddings
    and returns the closest match name (if below threshold).
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate embedding for input image
    input_embedding = embedder.embeddings([image])[0]

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


# Example usage
test_image = os.path.join(os.path.dirname(__file__), "faces", "test.jpg")
result = recognize_face(test_image)
print("🧾 Recognized as:", result)
