#!/usr/bin/env python3

import argparse
import json
import os
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract facial embeddings from an image using FaceNet (facenet-pytorch).")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default=None, help="Optional path to save embeddings (.npy or .json)")
    parser.add_argument("--all", action="store_true", help="Return embeddings for all detected faces. If not set, returns largest face only.")
    parser.add_argument("--model", default="vggface2", choices=["vggface2", "casia-webface"], help="Pretrained FaceNet weights")
    parser.add_argument("--min-face-size", type=int, default=40, help="Minimum face size for detection")
    parser.add_argument("--device", default="auto", help="'cpu', 'cuda', or 'auto' (default)")
    return parser.parse_args()


def select_largest_box(boxes: np.ndarray) -> Optional[int]:
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


essential_keys = ["num_faces", "embedding_dim", "embeddings"]


def save_embeddings(output_path: str, embeddings: List[np.ndarray]) -> None:
    if output_path.lower().endswith(".npy"):
        emb_array = np.stack(embeddings, axis=0)
        np.save(output_path, emb_array)
    else:
        serializable = [emb.tolist() for emb in embeddings]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"embeddings": serializable}, f)


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=args.min_face_size)
    resnet = InceptionResnetV1(pretrained=args.model).eval().to(device)

    image = Image.open(args.image).convert("RGB")

    # Detect boxes first for control over which faces to process
    boxes, probs = mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("No faces detected in the provided image.")

    if not args.all:
        idx = select_largest_box(np.array(boxes))
        if idx is None:
            raise RuntimeError("Failed to select any face for embedding.")
        boxes = np.array([boxes[idx]])

    # Align and crop faces to 160x160 using MTCNN's extract
    faces_aligned = mtcnn.extract(image, boxes, save_path=None)
    if faces_aligned is None or (isinstance(faces_aligned, list) and len(faces_aligned) == 0):
        raise RuntimeError("Failed to align faces for embedding.")

    if isinstance(faces_aligned, list):
        batch = torch.stack(faces_aligned).to(device)
    else:
        # faces_aligned is already a tensor of shape [N, 3, 160, 160]
        batch = faces_aligned.to(device)

    with torch.no_grad():
        emb_t = resnet(batch)
        embeddings = emb_t.cpu().numpy().astype(np.float32)

    print(json.dumps({
        "num_faces": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "embeddings": [emb.tolist() for emb in embeddings]
    }))

    if args.output:
        save_embeddings(args.output, [e for e in embeddings])


if __name__ == "__main__":
    main()
