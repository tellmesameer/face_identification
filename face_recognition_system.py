import os
import shutil
import pickle
import hashlib
from pathlib import Path

import cv2
import faiss
import numpy as np
import face_recognition


class FaceRecognitionSystem:
    def __init__(self, base_path="."):
        self.base_path = Path(base_path)
        self.cache_dir = self.base_path / "cache" / "embeddings"
        self.storage_dir = self.base_path / "storage"
        self.model_dir = self.base_path / "models"

        self.setup_directories()

        self.embedding_dim = 128
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []

        self.load_dnn_model()

    def setup_directories(self):
        dirs = [
            self.base_path / "data" / "reference_images",
            self.base_path / "data" / "results",
            self.cache_dir,
            self.storage_dir,
            self.model_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def load_dnn_model(self):
        proto = str(self.model_dir / "deploy.prototxt")
        model = str(self.model_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel")

        if not os.path.exists(proto) or not os.path.exists(model):
            raise FileNotFoundError("DNN model files missing in /models")

        self.net = cv2.dnn.readNetFromCaffe(proto, model)

    def _stable_cache_file(self, image_path):
        key = hashlib.sha1(str(image_path).encode()).hexdigest()
        return self.cache_dir / f"{key}.pkl"

    def _load_rgb(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def detect_faces(self, image):
        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")

                boxes.append((y1, x2, y2, x1))

        return boxes

    def _get_embedding(self, image_path):
        cache_file = self._stable_cache_file(image_path)

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except:
                pass

        encodings = []
        try:
            rgb = self._load_rgb(image_path)
            boxes = self.detect_faces(rgb)

            if boxes:
                encodings = face_recognition.face_encodings(rgb, boxes)
            else:
                print(f"No face: {image_path}")

        except Exception as e:
            print(f"Error {image_path}: {e}")

        if encodings:
            with open(cache_file, "wb") as f:
                pickle.dump(encodings, f)

        return encodings

    def process_dataset(self, dataset_path):
        dataset_dir = Path(dataset_path)
        print(f"Processing dataset: {dataset_dir}")

        valid_ext = {".jpg", ".jpeg", ".png", ".webp"}

        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if Path(file).suffix.lower() not in valid_ext:
                    continue

                path = os.path.join(root, file)
                print(f"Loading {path}")

                encs = self._get_embedding(path)

                for enc in encs:
                    self.index.add(np.array([enc], dtype=np.float32))
                    self.metadata.append(path)

        print(f"Indexed faces: {len(self.metadata)}")

    def _compute_centroid(self, ref_path, max_ref=10):
        encodings = []
        ref_dir = Path(ref_path)

        count = 0
        for file in sorted(os.listdir(ref_dir)):
            if count >= max_ref:
                break

            path = ref_dir / file
            if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue

            print(f"Ref: {path}")
            faces = self._get_embedding(path)

            if faces:
                encodings.append(faces[0])
                count += 1

        if not encodings:
            raise ValueError("No valid reference faces")

        centroid = np.mean(encodings, axis=0)
        return np.array([centroid], dtype=np.float32)

    def find_matches(self, ref_path, threshold=0.6, max_ref=10, max_results=10):
        query = self._compute_centroid(ref_path, max_ref)

        if len(self.metadata) == 0:
            print("No dataset indexed")
            return []

        D, I = self.index.search(query, len(self.metadata))

        results = []
        seen = set()

        for dist, idx in zip(D[0], I[0]):
            if dist <= threshold * threshold:
                path = self.metadata[idx]
                if path not in seen:
                    seen.add(path)
                    results.append(path)

                if len(results) >= max_results:
                    break

        print(f"Matches: {len(results)}")
        return results

    def copy_results(self, results, out_path):
        out = Path(out_path)
        out.mkdir(parents=True, exist_ok=True)

        for p in results:
            name = Path(p).name
            dest = out / name

            c = 1
            while dest.exists():
                dest = out / f"{dest.stem}_{c}{dest.suffix}"
                c += 1

            shutil.copy2(p, dest)

        print("Copied results")


if __name__ == "__main__":
    sys = FaceRecognitionSystem(base_path=r"e:\face_identification")

    sys.process_dataset(r"D:\nandalal_photos")

    results = sys.find_matches(
        r"e:\face_identification\data\reference_images",
        max_ref=10,
        max_results=10
    )

    sys.copy_results(results, r"e:\face_identification\data\results")