import os
import json
from pathlib import Path
from typing import List

from .config import (
    DEFAULT_THRESHOLD,
    DEFAULT_DEST,
    UNCATEGORISED_LABEL,
)
from .io_utils import (
    list_all_files,
    extract_text_from_file,
    move_file_to_category,
    load_or_initialize_labels,
    append_to_labels_json,
)
from .model_utils import KNNModelWrapper


def run_organiser(
    source: Path,
    dest: Path,
    threshold: float = DEFAULT_THRESHOLD,
    dry_run: bool = False,
    retrain: bool = False,
) -> None:
    """
    1. Load or initialize (examples, labels).
    2. Train (if --retrain) or load existing KNN model (with embeddings and metadata).
    3. Scan ALL files under `source` and compute (predicted_label, mean_distance) for each.
       - If mean_distance <= threshold: move immediately to <dest>/<predicted_label>
       - If mean_distance > threshold: collect into to_label list (DO NOT move yet)
    4. Once the confident files are moved, prompt you to label each file in to_label (while it still resides under source):
       - If you type a new label: move that file from source → dest/<new_label> and append to labels.json
       - If you press Enter (skip): move that file from source → dest/Uncategorised
    """

    # 1. Load or initialize labels
    examples, labels = load_or_initialize_labels()

    knn_wrapper = KNNModelWrapper()
    model_exists = Path("knn_model.joblib").exists()

    # 2. Train or load
    if retrain or not model_exists:
        if not examples or not labels:
            raise RuntimeError("No training examples found (labels.json or training_labels.json).")
        print(f"[INFO] Training new KNN model on {len(examples)} examples...")
        knn_wrapper.train(examples, labels)
        # This will write out:
        #   - knn_model.joblib
        #   - embeddings.npy
        #   - model_metadata.json
        knn_wrapper.save()
    else:
        print("[INFO] Loading existing KNN model from disk...")
        # This load() must now read both knn_model.joblib AND embeddings.npy (Option 1 change).
        knn_wrapper.load()

    # 3. Scan and classify (but do NOT move low-confidence yet)
    all_files = list_all_files(source)
    print(f"[INFO] Found {len(all_files)} files under {source}.")

    confident_moves: List[(Path, str)] = []
    to_label: List[Path] = []

    for file_path in all_files:
        text = extract_text_from_file(file_path)
        predicted_label, mean_distance = knn_wrapper.predict_with_confidence(text)

        if mean_distance > threshold:
            # collect in to_label (do NOT move yet)
            to_label.append(file_path)
        else:
            # confident → move immediately
            confident_moves.append((file_path, predicted_label))

    # 4. Move all the confidently classified files now
    for file_path, category in confident_moves:
        if dry_run:
            print(f"[DRY-RUN] {file_path.name} → [{category}]")
        else:
            move_file_to_category(file_path, dest, category)

    # 5. Handle the “uncategorised” list
    if to_label:
        print(f"\n[INFO] {len(to_label)} file(s) need manual labeling.")
        resp = input("Would you like to label them now? [y/N]: ").strip().lower()
        if resp == "y":
            for file_path in to_label:
                print(f"\nFile: {file_path.name}")
                new_label = input("  Enter a label (or press Enter to skip → send to 'Uncategorised'): ").strip()
                if new_label:
                    # Move from source → dest/<new_label> and save to labels.json
                    if dry_run:
                        print(f"  [DRY-RUN] {file_path.name} → [{new_label}]")
                    else:
                        move_file_to_category(file_path, dest, new_label)
                        append_to_labels_json(file_path.name, new_label)
                else:
                    # Move from source → dest/Uncategorised
                    if dry_run:
                        print(f"  [DRY-RUN] {file_path.name} → [{UNCATEGORISED_LABEL}]")
                    else:
                        move_file_to_category(file_path, dest, UNCATEGORISED_LABEL)
        else:
            # User chose not to label—send all to Uncategorised
            for file_path in to_label:
                if dry_run:
                    print(f"[DRY-RUN] {file_path.name} → [{UNCATEGORISED_LABEL}]")
                else:
                    move_file_to_category(file_path, dest, UNCATEGORISED_LABEL)
    else:
        print("[INFO] No files needed manual labeling.")

    print("[INFO] Done.")
