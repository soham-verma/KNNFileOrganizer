import pytest
from knn_file_organiser.model_utils import train_knn, predict_label

@pytest.fixture
def simple_seed():
    examples = ["bank statement April", "passport scanned ID", "university transcript"]
    labels = ["Finance", "Identification", "Education"]
    return examples, labels

def test_knn_predict_correct_label(tmp_path, simple_seed):
    examples, labels = simple_seed
    knn = train_knn(examples, labels)
    # ensure basic texts map to correct labels:
    assert predict_label(knn, "recent bank statement") == "Finance"
    assert predict_label(knn, "scan of passport") == "Identification"
    assert predict_label(knn, "degree transcript PDF") == "Education"
