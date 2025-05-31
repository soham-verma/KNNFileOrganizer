import PyPDF2
import pytest
from pathlib import Path
from knn_file_organiser.io_utils import extract_text_from_file

@pytest.fixture
def create_pdf(tmp_path):
    pdf_path = tmp_path / "dummy.pdf"
    writer = PyPDF2.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    # Overwrite with some text? If PyMuPDF is used, create a real PDF with text.
    return pdf_path

def test_extract_text_from_pdf(create_pdf):
    text = extract_text_from_file(create_pdf)
    assert isinstance(text, str)
    # If your extract_text returns empty for a blank PDF, ensure it doesn't error out.
    assert text == "" or isinstance(text, str)

def test_extract_text_from_txt(tmp_path):
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("Hello World")
    assert extract_text_from_file(txt_path) == "Hello World"
