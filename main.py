import os
import shutil
import time
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
import fitz  # PyMuPDF

LABELS_FILE = "labels.json"
TRAINING_LABELS_FILE = "training_labels.json"
UNCATEGORIZED = "Uncategorized"

# -------------------------------
# Load or initialize label memory
# -------------------------------
def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            data = json.load(f)
            return data["examples"], data["labels"]
    else:
        return [], []

def save_labels(examples, labels):
    with open(LABELS_FILE, "w") as f:
        json.dump({"examples": examples, "labels": labels}, f, indent=2)

def load_pretrained():
    try:
        with open(TRAINING_LABELS_FILE, "r") as f:
            data = json.load(f)
            return data["examples"], data["labels"]
    except FileNotFoundError:
        print("⚠️ No training_labels.json found.")
        return [], []

# -------------------------------
# User-defined pre-training prompt
# -------------------------------
def ask_user_for_initial_labels():
    print("Let's help the model with a few categories!")
    input_labels = input("Enter possible categories (comma-separated):\n> ").split(',')
    examples = []
    labels = []

    for label in [lbl.strip().capitalize() for lbl in input_labels if lbl.strip()]:
        example = input(f"Give an example description for '{label}' (or press enter to skip):\n> ").strip()
        if example:
            examples.append(example)
            labels.append(label)
    return examples, labels

# -------------------------------
# Load all training examples
# -------------------------------
examples, labels = load_labels()

# Load pretrained file if no prior training
if not examples:
    print("Loading pretrained examples...")
    pre_examples, pre_labels = load_pretrained()
    examples += pre_examples
    labels += pre_labels

# Ask user for label guidance
user_examples, user_labels = ask_user_for_initial_labels()
examples += user_examples
labels += user_labels

# -------------------------------
# Load model and train classifier
# -------------------------------
print("Loading model and training classifier...")
start_time = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')

classifier = None
if examples:
    X_train = model.encode(examples)
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(X_train, labels)

print(f"✅ Model ready. {len(set(labels))} categories known.\n")

# -------------------------------
# Extract text from PDFs
# -------------------------------
def extract_text(file_path):
    try:
        if file_path.suffix.lower() == '.pdf':
            text = ""
            doc = fitz.open(str(file_path))
            for page in doc:
                text += page.get_text()
            return text.strip()
    except Exception as e:
        print(f"⚠️ Error reading {file_path.name}: {e}")
        return ""
    return ""

# -------------------------------
# Predict label
# -------------------------------
def predict_label(text):
    if classifier:
        embedding = model.encode([text])
        dist, idx = classifier.kneighbors(embedding, return_distance=True)
        if dist[0][0] < 0.6:  # confident threshold
            return classifier.predict(embedding)[0]
    return None

# -------------------------------
# Move file to destination folder
# -------------------------------
def move_to_folder(file_path, dest_root, category):
    dest_folder = dest_root / category
    dest_folder.mkdir(parents=True, exist_ok=True)
    shutil.move(str(file_path), dest_folder / file_path.name)
    print(f"→ {file_path.name} → [{category}]")

# -------------------------------
# Review uncategorized files
# -------------------------------
def review_uncategorized(dest_root):
    uncategorized_folder = dest_root / UNCATEGORIZED
    if not uncategorized_folder.exists():
        return

    files = list(uncategorized_folder.glob("*.*"))
    if not files:
        return

    print(f"\n{len(files)} file(s) in '{UNCATEGORIZED}'. Review now? (y/n)")
    if input(">").lower() != 'y':
        return

    for file_path in files:
        print(f"\n{file_path.name}")
        print("Existing categories:")
        for i, label in enumerate(sorted(set(labels))):
            print(f"  [{i}] {label}")
        choice = input("Enter number or new category: ").strip()
        if choice.isdigit() and int(choice) < len(set(labels)):
            label = sorted(set(labels))[int(choice)]
        else:
            label = choice.capitalize()

        # Move and re-train
        move_to_folder(file_path, dest_root, label)
        full_text = extract_text(file_path)
        input_text = f"{file_path.stem} {full_text}"
        examples.append(input_text)
        labels.append(label)

    save_labels(examples, labels)
    print("Updated model knowledge with your choices.")

# -------------------------------
# Main organize function
# -------------------------------
def organize(path):
    global classifier
    base = Path(path).expanduser()
    dest_root = base / "Organized"
    dest_root.mkdir(parents=True, exist_ok=True)

    files = [f for f in base.iterdir() if f.is_file()]
    print(f"Scanning: {base.resolve()}")
    print("--------------------------------------------------")

    for file_path in files:
        print(f"\n📁 Processing: {file_path.name}")
        full_text = extract_text(file_path)
        input_text = f"{file_path.stem} {full_text}"
        label = predict_label(input_text)

        if label:
            move_to_folder(file_path, dest_root, label)
        else:
            move_to_folder(file_path, dest_root, UNCATEGORIZED)

    elapsed = time.time() - start_time
    print(f"\n✅ Finished organizing in {elapsed:.2f}s")
    review_uncategorized(dest_root)

# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    target = "~/Desktop/TestFiles"
    if not target:
        target = "~/Desktop/TestFiles"
    organize(target)
