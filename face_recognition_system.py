import os
import pickle
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import face_recognition


class FaceRecognitionSystem:
    def __init__(self, base_path="."):
        self.base_path = Path(base_path)
        self.cache_dir = self.base_path / "cache" / "embeddings"
        self.storage_dir = self.base_path / "storage"
        self.model_dir = self.base_path / "models"
        self.results_dir = self.base_path / "data" / "results"
        self.valid_ext = {".jpg", ".jpeg", ".png", ".webp"}

        self.setup_directories()

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

    def _normalize_path(self, path):
        return Path(path).as_posix()

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

    def detect_faces(self, image, confidence_threshold=0.4, min_face_size=40):
        (h, w) = image.shape[:2]
        resized = image
        scale = 1.0

        downscale = 600 / max(h, w)
        if downscale < 1:
            resized = cv2.resize(image, (int(w * downscale), int(h * downscale)))
            scale = 1 / downscale

        (rh, rw) = resized.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(resized, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([rw, rh, rw, rh])
                x1, y1, x2, y2 = box.astype("int")

                x1 = max(0, min(rw - 1, x1))
                y1 = max(0, min(rh - 1, y1))
                x2 = max(0, min(rw, x2))
                y2 = max(0, min(rh, y2))

                if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
                    continue

                if scale != 1.0:
                    x1 = int(x1 * scale)
                    y1 = int(y1 * scale)
                    x2 = int(x2 * scale)
                    y2 = int(y2 * scale)

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
                encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=1)
            else:
                print(f"No face: {image_path}")

        except Exception as e:
            print(f"Error {image_path}: {e}")

        if encodings:
            with open(cache_file, "wb") as f:
                pickle.dump(encodings, f)

        return encodings

    def iter_dataset_files(self, dataset_path):
        dataset_dir = Path(dataset_path)
        print(f"Processing dataset: {dataset_dir}")

        for root, _, files in os.walk(dataset_dir):
            for file in sorted(files):
                path = Path(root) / file
                if path.suffix.lower() not in self.valid_ext:
                    continue
                yield path

    def _get_reference_embeddings(self, ref_path):
        encodings = []
        ref_dir = Path(ref_path)

        for path in sorted(ref_dir.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in self.valid_ext:
                continue

            print(f"Ref: {path}")
            faces = self._get_embedding(path)

            if faces:
                encodings.append(faces[0])

        if not encodings:
            raise ValueError("No valid reference faces")

        return np.array(encodings, dtype=np.float32)

    def _compute_centroid(self, reference_embeddings):
        centroid = np.mean(reference_embeddings, axis=0)
        return np.array([centroid], dtype=np.float32)

    def _reference_signature(self, ref_path):
        ref_dir = Path(ref_path)
        digest = hashlib.sha1()

        for path in sorted(ref_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.valid_ext:
                continue
            stat = path.stat()
            digest.update(self._normalize_path(path).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))

        return digest.hexdigest()

    def _load_feedback_rules(self):
        feedback_file = self.base_path / "data" / "results" / "match_feedback.json"
        if not feedback_file.exists():
            return {}

        try:
            payload = json.loads(feedback_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

        positives = payload.get("training_candidates", {}).get("positive_matches", [])
        negatives = payload.get("training_candidates", {}).get("negative_matches", [])

        rules = {}

        positive_mean_distances = [
            item.get("mean_reference_distance")
            for item in positives
            if item.get("mean_reference_distance") is not None
        ]
        negative_mean_distances = [
            item.get("mean_reference_distance")
            for item in negatives
            if item.get("mean_reference_distance") is not None
        ]

        if positive_mean_distances and negative_mean_distances:
            max_positive_mean = max(positive_mean_distances)
            min_negative_mean = min(negative_mean_distances)
            if max_positive_mean < min_negative_mean:
                rules["max_mean_reference_distance"] = (
                    max_positive_mean + min_negative_mean
                ) / 2.0

        rules["excluded_image_paths"] = {
            item["image_path"]
            for item in negatives
            if item.get("image_path")
        }

        return rules

    def _analysis_paths(self):
        return {
            "records": self.results_dir / "analysis_records.jsonl",
            "state": self.results_dir / "analysis_state.json",
            "matches": self.results_dir / "matched_paths.json",
        }

    def _reset_analysis_outputs(self, paths):
        for key in ("records", "matches"):
            if paths[key].exists():
                paths[key].unlink()

    def _load_analysis_state(self, ref_path, threshold):
        paths = self._analysis_paths()
        signature = self._reference_signature(ref_path)
        state = {
            "reference_signature": signature,
            "threshold": threshold,
            "analyzed_paths": [],
            "processed_count": 0,
            "match_count": 0,
            "non_match_count": 0,
        }

        if paths["state"].exists():
            try:
                current = json.loads(paths["state"].read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}

            if (
                current.get("reference_signature") == signature
                and current.get("threshold") == threshold
            ):
                state.update(current)
            else:
                self._reset_analysis_outputs(paths)

        analyzed_paths = set(state.get("analyzed_paths", []))
        return state, analyzed_paths, paths

    def _save_analysis_state(self, state, paths):
        paths["state"].write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _append_analysis_record(self, record, paths):
        with open(paths["records"], "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_matches_snapshot(self, matches, paths):
        payload = {
            "match_count": len(matches),
            "matches": matches,
        }
        paths["matches"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _build_match_result(self, image_path, centroid_distance, min_reference_distance, mean_reference_distance):
        confidence = max(0.0, min(1.0, 1.0 - min_reference_distance))
        return {
            "image_path": self._normalize_path(image_path),
            "confidence": round(confidence, 4),
            "centroid_distance": round(float(centroid_distance), 4),
            "min_reference_distance": round(float(min_reference_distance), 4),
            "mean_reference_distance": round(float(mean_reference_distance), 4),
        }

    def _analyze_image(self, image_path, reference_embeddings, centroid, threshold, feedback_rules):
        encodings = self._get_embedding(image_path)
        normalized_path = self._normalize_path(image_path)

        if normalized_path in feedback_rules.get("excluded_image_paths", set()):
            return {
                "image_path": normalized_path,
                "status": "skipped_feedback_exclusion",
                "is_match": False,
            }

        if not encodings:
            return {
                "image_path": normalized_path,
                "status": "no_face",
                "is_match": False,
            }

        best_match = None
        best_min_reference_distance = None

        for enc in encodings:
            candidate_embedding = np.array(enc, dtype=np.float32)
            reference_distances = np.linalg.norm(reference_embeddings - candidate_embedding, axis=1)
            min_reference_distance = float(np.min(reference_distances))
            mean_reference_distance = float(np.mean(reference_distances))

            if min_reference_distance > threshold:
                continue

            max_mean_reference_distance = feedback_rules.get("max_mean_reference_distance")
            if (
                max_mean_reference_distance is not None
                and mean_reference_distance > max_mean_reference_distance
            ):
                continue

            centroid_distance = float(np.linalg.norm(candidate_embedding - centroid))
            match = self._build_match_result(
                image_path,
                centroid_distance,
                min_reference_distance,
                mean_reference_distance,
            )

            if (
                best_match is None
                or min_reference_distance < best_min_reference_distance
            ):
                best_match = match
                best_min_reference_distance = min_reference_distance

        if best_match is not None:
            return {
                **best_match,
                "status": "match",
                "is_match": True,
            }

        return {
            "image_path": normalized_path,
            "status": "not_match",
            "is_match": False,
        }

    def find_matches(self, ref_path, dataset_path, threshold=0.5):
        reference_embeddings = self._get_reference_embeddings(ref_path)
        centroid = self._compute_centroid(reference_embeddings)[0]
        feedback_rules = self._load_feedback_rules()
        state, analyzed_paths, paths = self._load_analysis_state(ref_path, threshold)

        matches = []
        if paths["records"].exists():
            with open(paths["records"], "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("is_match"):
                        matches.append({
                            "image_path": record["image_path"],
                            "confidence": record.get("confidence"),
                            "centroid_distance": record.get("centroid_distance"),
                            "min_reference_distance": record.get("min_reference_distance"),
                            "mean_reference_distance": record.get("mean_reference_distance"),
                        })

        for path in self.iter_dataset_files(dataset_path):
            normalized_path = self._normalize_path(path)
            if normalized_path in analyzed_paths:
                continue

            print(f"Analyzing {path}")
            record = self._analyze_image(
                path,
                reference_embeddings,
                centroid,
                threshold,
                feedback_rules,
            )

            self._append_analysis_record(record, paths)
            analyzed_paths.add(normalized_path)
            state["analyzed_paths"] = sorted(analyzed_paths)
            state["processed_count"] = int(state.get("processed_count", 0)) + 1

            if record.get("is_match"):
                matches.append({
                    "image_path": record["image_path"],
                    "confidence": record.get("confidence"),
                    "centroid_distance": record.get("centroid_distance"),
                    "min_reference_distance": record.get("min_reference_distance"),
                    "mean_reference_distance": record.get("mean_reference_distance"),
                })
                state["match_count"] = int(state.get("match_count", 0)) + 1
            else:
                state["non_match_count"] = int(state.get("non_match_count", 0)) + 1

            self._write_matches_snapshot(matches, paths)
            self._save_analysis_state(state, paths)

        print(f"Matches: {len(matches)}")
        return matches

    def save_result_paths(self, results, out_path):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "match_count": len(results),
            "matches": results,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved result paths to {out}")


if __name__ == "__main__":
    sys = FaceRecognitionSystem(base_path=r"e:\face_identification")

    results = sys.find_matches(
        r"e:\face_identification\data\reference_images",
        r"D:\nandalal_photos",
    )

    sys.save_result_paths(results, r"e:\face_identification\data\results\matched_paths.json")
