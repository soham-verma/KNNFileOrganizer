import shutil
import pytest
from pathlib import Path
from knn_file_organiser.organiser import run_organiser

@pytest.fixture
def sample_files(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    # Create simple text files named so they map to your seed labels
    (src / "bank_doc.txt").write_text("bank statement April")
    (src / "id_scan.txt").write_text("driver license scan")
    return src

def test_run_organiser_dry_run(tmp_path, monkeypatch, sample_files):
    # monkeypatch the model to always return "Finance" for any file
    from knn_file_organiser import model_utils
    class DummyKNN:
        def predict(self, vecs): return ["Finance"]
    monkeypatch.setattr(model_utils, "load_or_train_model", lambda *args, **kwargs: DummyKNN())

    dest = tmp_path / "organised"
    run_organiser(source=sample_files, dest=dest, threshold=0.5, dry_run=True)
    # In dry_run mode, files should NOT be moved; dest folder should either not exist
    assert not dest.exists()
