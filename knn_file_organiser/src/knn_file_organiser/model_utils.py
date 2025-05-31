import os
import json
import joblib
import numpy as np

from pathlib import Path
from typing import List, Tuple, Optional

from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer

from .config import (
    LABELS_FILE,
    TRAINING_LABELS_FILE,
    MODEL_FILE,
    EMBEDDINGS_FILE,
    KNN_NEIGHBORS,
)


class KNNModelWrapper:
    """
    Wraps a SentenceTransformer embedding model + a KNN classifier.
    Provides train, predict, save, and load functionality.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.knn: Optional[KNeighborsClassifier] = None
        self.examples: List[str] = []
        self.labels: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def train(self, examples: List[str], labels: List[str]) -> None:
        """
        Given `examples` (texts/filenames) and `labels`, compute embeddings and train KNN.
        """
        if not examples or not labels or len(examples) != len(labels):
            raise ValueError("Examples and labels must be non-empty and of the same length.")

        self.examples = examples
        self.labels = labels
        # Compute embeddings matrix of shape (n_examples, embedding_dim)
        self.embeddings = self.embedder.encode(examples, convert_to_numpy=True)
        # Initialize and train KNN
        self.knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
        self.knn.fit(self.embeddings, labels)

    def predict_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Returns (predicted_label, mean_distance_to_neighbors).
        Uses `kneighbors` to get distances and labels of nearest neighbors.
        """
        if self.knn is None or self.embeddings is None:
            raise RuntimeError("Model has not been trained or loaded.")

        vec = self.embedder.encode([text], convert_to_numpy=True)
        distances, indices = self.knn.kneighbors(vec, n_neighbors=KNN_NEIGHBORS)
        # distances shape: (1, k), indices shape: (1, k)
        distances = distances[0]  # (k,)
        indices = indices[0]      # (k,)
        # majority vote for label among neighbors
        neighbor_labels = [self.labels[i] for i in indices]
        # simple majority vote:
        label = max(set(neighbor_labels), key=neighbor_labels.count)
        # measure confidence as average distance
        mean_dist = float(np.mean(distances))
        return label, mean_dist

    def save(self, model_path: Path = Path(MODEL_FILE), embeddings_path: Path = Path(EMBEDDINGS_FILE)) -> None:
        """
        Persist the trained KNN model and the embeddings/labels to disk.
        """
        if self.knn is None or self.embeddings is None:
            raise RuntimeError("Nothing to save; model is not trained.")

        # Save KNN classifier (which includes examples order implicitly via embeddings)
        joblib.dump(self.knn, model_path)
        # Save embeddings so that we can potentially inspect them later
        np.save(embeddings_path, self.embeddings)
        # Save examples & labels (in case we need to reload them)
        meta = {"examples": self.examples, "labels": self.labels}
        Path("model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, model_path: Path = Path(MODEL_FILE), embeddings_path: Path = Path(EMBEDDINGS_FILE)) -> None:
      if not model_path.exists():
          raise FileNotFoundError(f"No saved model found at {model_path}.")
      self.knn = joblib.load(model_path)

      if embeddings_path.exists():
          self.embeddings = np.load(embeddings_path)
      else:
          raise FileNotFoundError(f"No embeddings file found at {embeddings_path}.")

      meta_path = Path("model_metadata.json")
      if meta_path.exists():
          meta = json.loads(meta_path.read_text(encoding="utf-8"))
          self.examples = meta.get("examples", [])
          self.labels   = meta.get("labels", [])
      else:
          raise FileNotFoundError("model_metadata.json is missing; cannot load labels/examples.")



    def is_trained(self) -> bool:
        return self.knn is not None

