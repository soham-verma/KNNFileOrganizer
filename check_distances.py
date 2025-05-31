# check_distances.py

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

# === 1) Load your training seeds ===
training_path = Path("training_labels.json")
if not training_path.exists():
    print("ERROR: training_labels.json not found in this folder.")
    exit(1)

training_data = json.loads(training_path.read_text(encoding="utf-8"))
examples = training_data["examples"]
labels = training_data["labels"]

# === 2) Initialize the same embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")
# Compute embeddings for every seed example
train_embeds = model.encode(examples, convert_to_numpy=True)

# === 3) Gather all filenames under your TestFiles folder ===
test_dir = Path.home() / "Desktop" / "TestFiles"
if not test_dir.exists() or not test_dir.is_dir():
    print(f"ERROR: {test_dir} does not exist or is not a directory.")
    exit(1)

filenames = [p.name for p in test_dir.iterdir() if p.is_file()]

# === 4) Normalize each filename exactly as your code does ===
def normalize_name(fn: str) -> str:
    base = Path(fn).stem  # strips “.pdf”, “.txt”, etc.
    # Replace underscores, hyphens, ampersands, plus signs with spaces
    temp = re.sub(r"[_\-\&\+]+", " ", base)
    # Remove any other punctuation (keep only alphanumeric + whitespace)
    temp = re.sub(r"[^\w\s]", " ", temp)
    # Collapse multiple spaces into one, then lowercase
    return re.sub(r"\s+", " ", temp).strip().lower()

normalized = [normalize_name(fn) for fn in filenames]

# === 5) Embed all normalized test filenames ===
test_embeds = model.encode(normalized, convert_to_numpy=True)

# === 6) Compute cosine distances between every test embed and every seed embed ===
dist_matrix = cosine_distances(test_embeds, train_embeds)

# === 7) Build a results table: nearest example + label + distance for each file ===
results = []
for i, fn in enumerate(filenames):
    distances = dist_matrix[i]
    nearest_idx = int(np.argmin(distances))
    results.append({
        "Filename": fn,
        "Normalized": normalized[i],
        "Nearest Seed": examples[nearest_idx],
        "Seed Label": labels[nearest_idx],
        "Distance": float(distances[nearest_idx])
    })

# === 8) Show a sorted table (lowest distance first) ===
df = pd.DataFrame(results).sort_values("Distance")
pd.set_option("display.max_colwidth", 40)
print("\nNearest‐Neighbor Distances (sorted):\n")
print(df.to_string(index=False))
