# Revised README content without code blocks to avoid output cut-off
readme_content = """
# 📁 AI-Powered File Organizer (Offline + Customizable)

An intelligent, offline-first file organizer built with Python + SentenceTransformers. It automatically predicts appropriate folder names for your files based on filename and content, organizes them into labeled directories, and improves over time through learning.

> 🧠 It learns from you — so you only have to review *unrecognized files once*. Future runs are automatic.

---

## 🚀 Features

- ✅ Offline AI classification (no internet needed)
- 📂 Organizes files by predicting folders (Education, Finance, ID, etc.)
- 🔍 Scans PDFs and filenames for better accuracy
- 🧠 Learns from your corrections and saves them
- 🧑‍🏫 Optional: give it training data or guide it during the first run

---

## 🛠 Requirements

- Python 3.8+
- Install dependencies:
  pip install -r requirements.txt

**Dependencies:**

- sentence-transformers
- scikit-learn
- PyMuPDF

---

## 📦 Installation
`
  cd ai-file-organizer
  python3 main.py
`
---

## 🧠 How It Works

1. Loads training examples from:
   - labels.json (your saved knowledge)
   - training_labels.json (pre-seeded examples)
   - User-defined labels (optional prompt at start)

2. Scans a directory (default: ~/Desktop/TestFiles)

3. Predicts folder/category for each file based on name + content

4. Moves files into:
   - Organized/<Predicted Category>/
   - Or Organized/Uncategorized/ if unsure

5. Asks if you'd like to review the uncategorized files to help it learn

---

## 📁 Directory Example

After organizing:

Organized/
├── Education/
│   └── UOW Transcript.pdf
├── Finance/
│   └── bank_statement.pdf
├── Health/
│   └── Medibank Policy.pdf
├── Uncategorized/
│   └── unknown_file.pdf

---

## 🧪 Customizing

### Add Your Own Training Examples

Edit training_labels.json to preload examples and categories.

{
  "examples": [
    "University transcript",
    "Medibank insurance card",
    "Visa approval grant letter"
  ],
  "labels": [
    "Education",
    "Health",
    "Immigration"
  ]
}

### Change File Target Path

At the bottom of main.py:
  `target = "~/TestFiles"`

Change it to wherever you want to scan files from.

---

## 🔄 Example Use Cases

- Decluttering your Downloads/ folder
- Organizing scanned documents
- Sorting personal PDFs, bank records, IDs, etc.

---


## 📄 License

MIT License © 2025  
Built with ❤️ by [Soham Verma](https://github.com/soham-verma)
"""

# Save to a markdown file
`
readme_path = "/mnt/data/README.md"
with open(readme_path, "w") as f:
    f.write(readme_content)

readme_path
`