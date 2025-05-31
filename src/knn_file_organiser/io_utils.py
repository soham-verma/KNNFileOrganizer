import os
import shutil
import re
import json
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from .config import LABELS_FILE, TRAINING_LABELS_FILE, UNCATEGORISED_LABEL


def list_all_files(source: Path) -> List[Path]:
    """
    Recursively list all files (not directories) under `source`.
    """
    return [p for p in source.rglob("*") if p.is_file()]


def extract_text_from_file(file_path: Path) -> str:
    """
    Extract text for embedding:
      1) If it's a PDF, try to pull real text from the first few pages.
      2) Otherwise (or if PDF text is empty), take the filename stem,
         remove punctuation/underscores/hyphens, and lowercase it.
    """
    # === 1) PDF text extraction (if applicable) ===
    try:
        if file_path.suffix.lower() == ".pdf":
            doc = fitz.open(str(file_path))
            text_chunks = []
            max_pages = min(3, doc.page_count)
            for i in range(max_pages):
                page = doc.load_page(i)
                text_chunks.append(page.get_text())
            joined = " ".join(text_chunks).strip()
            if joined:  # if we actually found text in the PDF
                return joined.lower()
    except Exception:
        pass  # if anything goes wrong, fallback to filename normalization below

    # === 2) Filename normalization ===
    # Example: "Medibank_Policy_Notification-1.pdf" → 
    #   "medibank policy notification 1"
    base = file_path.stem  # strips away ".pdf", ".txt", etc.

    # Replace any underscores, hyphens, ampersands, plus signs, etc., with spaces
    temp = re.sub(r"[_\-\&\+]+", " ", base)

    # Remove any remaining punctuation (keep letters, numbers, whitespace)
    temp = re.sub(r"[^\w\s]", " ", temp)

    # Collapse multiple spaces into a single space
    temp = re.sub(r"\s+", " ", temp).strip()

    # Finally, lowercase everything
    return temp.lower()




def move_file_to_category(file_path: Path, dest_root: Path, category: str) -> None:
    """
    Move `file_path` into `dest_root / category / <original-filename>`.
    Creates directories as needed.
    """
    target_dir = dest_root / category
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(file_path), str(target_dir / file_path.name))
    except Exception as e:
        print(f"[ERROR] Unable to move {file_path} → {target_dir}: {e}")


def load_or_initialize_labels() -> (List[str], List[str]):
    """
    Load examples/labels from LABELS_FILE (labels.json). If it doesn't exist or is empty,
    load from the TRAINING_LABELS_FILE (training_labels.json) instead.
    Returns (examples, labels) as two parallel lists.
    """
    examples, labels = [], []
    if Path(LABELS_FILE).exists():
        try:
            data = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
            examples = data.get("examples", [])
            labels = data.get("labels", [])
        except Exception:
            examples, labels = [], []

    if not examples or not labels:
        # Fallback to training set
        if Path(TRAINING_LABELS_FILE).exists():
            data = json.loads(Path(TRAINING_LABELS_FILE).read_text(encoding="utf-8"))
            examples = data.get("examples", [])
            labels = data.get("labels", [])
        else:
            examples, labels = [], []

    return examples, labels


def append_to_labels_json(filename: str, label: str) -> None:
    """
    Append a single (filename → label) mapping to LABELS_FILE. If the file doesn't exist,
    it will be created with the appropriate JSON structure.
    """
    if Path(LABELS_FILE).exists():
        data = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
    else:
        data = {"examples": [], "labels": []}

    data["examples"].append(filename)
    data["labels"].append(label)
    Path(LABELS_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")
