from keras_facenet import FaceNet
import cv2
import numpy as np

embedder = FaceNet()

image_path = 'faces/morgan.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

embeddings = embedder.embeddings([image])


embedding_vector = embeddings[0]


print("128D Face Embedding:")
print(embedding_vector)
print("Embedding shape:", embedding_vector.shape)
