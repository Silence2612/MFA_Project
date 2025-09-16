import os
import cv2
from keras_facenet import FaceNet

embedder = FaceNet()

# Dynamically set the image path
image_path = os.path.join(os.path.dirname(__file__), 'faces', 'morgan.jpg')

# Load image with error check
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

embeddings = embedder.embeddings([image])
embedding_vector = embeddings[0]

print("512D Face Embedding:")
print(embedding_vector)
print("Embedding shape:", embedding_vector.shape)
