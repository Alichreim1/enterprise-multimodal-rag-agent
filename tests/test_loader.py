from pathlib import Path

import pytest

from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.loader_factory import LoaderFactory
from rag.models.document import Document


def test_pdf_loader():
    file_path = Path(
        "data/documents/exactly_3_pages_page_2_empty.pdf"
    )

    loader = PDFLoader()

    documents = loader.load(file_path)

    assert isinstance(documents, list)
    assert len(documents) > 0

    for document in documents:
        assert isinstance(document, Document)

        assert document.content.strip()

        assert document.metadata["source"] == (
            "exactly_3_pages_page_2_empty.pdf"
        )

        assert document.metadata["file_type"] == "pdf"

        assert isinstance(
            document.metadata["page"],
            int,
        )

        assert document.metadata["page"] >= 1


def test_pdf_loader_preserves_page_numbers():
    file_path = Path(
        "data/documents/exactly_3_pages_page_2_empty.pdf"
    )

    loader = PDFLoader()

    documents = loader.load(file_path)

    page_numbers = [
        document.metadata["page"]
        for document in documents
    ]

    assert page_numbers == [1, 3]


def test_pdf_loader_skips_empty_pages():
    file_path = Path(
        "data/documents/exactly_3_pages_page_2_empty.pdf"
    )

    loader = PDFLoader()

    documents = loader.load(file_path)

    assert len(documents) == 2

    page_numbers = [
        document.metadata["page"]
        for document in documents
    ]

    assert page_numbers == [1, 3]

    assert documents[0].content.strip()
    assert documents[1].content.strip()


def test_loader_factory_returns_pdf_loader():
    file_path = Path(
        "data/documents/exactly_3_pages_page_2_empty.pdf"
    )

    loader = LoaderFactory.get_loader(file_path)

    assert isinstance(loader, PDFLoader)


def test_loader_factory_rejects_unsupported_file():
    file_path = Path("data/documents/example.csv")

    with pytest.raises(ValueError):
        LoaderFactory.get_loader(file_path)