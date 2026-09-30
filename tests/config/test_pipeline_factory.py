from pathlib import Path

import pytest

import config.bootstrap as bootstrap


class FakeTextPipeline:
    pass


class FakeMultimodalPipeline:
    pass


def test_build_pipeline_for_text_file(
    monkeypatch,
):

    expected_pipeline = FakeTextPipeline()

    called_with = {}

    def fake_build_rag_pipeline(file_path):

        called_with["file_path"] = file_path

        return expected_pipeline

    monkeypatch.setattr(
        bootstrap,
        "build_rag_pipeline",
        fake_build_rag_pipeline,
    )

    result = bootstrap.build_pipeline_for_file(
        "data/uploads/document.pdf"
    )

    assert result is expected_pipeline

    assert called_with["file_path"] == Path(
        "data/uploads/document.pdf"
    )


def test_build_pipeline_for_image_file(
    monkeypatch,
):

    expected_pipeline = FakeMultimodalPipeline()

    called_with = {}

    def fake_build_multimodal_pipeline(
        file_path,
    ):

        called_with["file_path"] = file_path

        return expected_pipeline

    monkeypatch.setattr(
        bootstrap,
        "build_multimodal_rag_pipeline",
        fake_build_multimodal_pipeline,
    )

    result = bootstrap.build_pipeline_for_file(
        "data/uploads/image.PNG"
    )

    assert result is expected_pipeline

    assert called_with["file_path"] == Path(
        "data/uploads/image.PNG"
    )


@pytest.mark.parametrize(
    "extension",
    [
        ".pdf",
        ".docx",
        ".txt",
    ],
)
def test_text_extensions_use_text_pipeline(
    monkeypatch,
    extension,
):

    text_pipeline = FakeTextPipeline()

    monkeypatch.setattr(
        bootstrap,
        "build_rag_pipeline",
        lambda file_path: text_pipeline,
    )

    result = bootstrap.build_pipeline_for_file(
        f"document{extension}"
    )

    assert result is text_pipeline


@pytest.mark.parametrize(
    "extension",
    [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    ],
)
def test_image_extensions_use_multimodal_pipeline(
    monkeypatch,
    extension,
):

    multimodal_pipeline = (
        FakeMultimodalPipeline()
    )

    monkeypatch.setattr(
        bootstrap,
        "build_multimodal_rag_pipeline",
        lambda file_path: multimodal_pipeline,
    )

    result = bootstrap.build_pipeline_for_file(
        f"image{extension}"
    )

    assert result is multimodal_pipeline


def test_unsupported_extension_raises_error():

    with pytest.raises(
        ValueError,
        match="Unsupported file type",
    ):
        bootstrap.build_pipeline_for_file(
            "archive.zip"
        )