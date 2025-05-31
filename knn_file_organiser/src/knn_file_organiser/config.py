from pathlib import Path

# Filenames for persisted data
LABELS_FILE = "labels.json"
TRAINING_LABELS_FILE = "training_labels.json"
MODEL_FILE = "knn_model.joblib"
EMBEDDINGS_FILE = "embeddings.npy"

# Special label for uncategorised files
UNCATEGORISED_LABEL = "Uncategorised"

# Defaults (can be overridden via CLI arguments)
DEFAULT_SOURCE = Path.home() / "Downloads"
DEFAULT_DEST = Path.cwd() / "Organised"
DEFAULT_THRESHOLD = 0.7  # If mean neighbour distance > threshold, mark as "Uncategorised"

# KNN hyperparameters
KNN_NEIGHBORS = 3
