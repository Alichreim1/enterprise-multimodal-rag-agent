from pathlib import Path

import pytest

from rag.loaders.loader_factory import LoaderFactory
from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.docx_loader import DOCXLoader
from rag.loaders.txt_loader import TXTLoader


def test_pdf_loader():

    loader = LoaderFactory.get_loader(
        Path("file.pdf")
    )

    assert isinstance(
        loader,
        PDFLoader,
    )


def test_docx_loader():

    loader = LoaderFactory.get_loader(
        Path("file.docx")
    )

    assert isinstance(
        loader,
        DOCXLoader,
    )


def test_txt_loader():

    loader = LoaderFactory.get_loader(
        Path("file.txt")
    )

    assert isinstance(
        loader,
        TXTLoader,
    )


def test_invalid_extension():

    with pytest.raises(ValueError):

        LoaderFactory.get_loader(
            Path("file.xyz")
        )