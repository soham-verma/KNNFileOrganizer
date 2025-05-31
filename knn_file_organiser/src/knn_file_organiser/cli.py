import argparse
from pathlib import Path
import sys

from .config import DEFAULT_SOURCE, DEFAULT_DEST, DEFAULT_THRESHOLD
from .organiser import run_organiser


def parse_args():
    parser = argparse.ArgumentParser(
        prog="knn-file-organiser",
        description="Organise files into folders using a KNN classifier on semantic embeddings."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Directory to scan for files (default: {DEFAULT_SOURCE})"
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination root for organised files (default: {DEFAULT_DEST})"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Distance threshold above which files are marked as uncategorised (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run: show where files would be placed without actually moving them."
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force re-training of the KNN model, even if a saved model exists."
    )
    parser.add_argument(
        "--version",
        action="version",
        version="knn-file-organiser 0.1.0"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    source = args.source.expanduser().resolve()
    dest = args.dest.expanduser().resolve()

    if not source.exists() or not source.is_dir():
        print(f"[ERROR] Source directory does not exist: {source}")
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)

    run_organiser(
        source=source,
        dest=dest,
        threshold=args.threshold,
        dry_run=args.dry_run,
        retrain=args.retrain,
    )


if __name__ == "__main__":
    main()
