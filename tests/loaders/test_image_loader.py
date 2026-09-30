from PIL import Image

from rag.loaders.image_loader import (
    ImageLoader,
)


def test_image_loader(
    tmp_path,
):

    image_path = (
        tmp_path / "chart.png"
    )

    image = Image.new(
        mode="RGB",
        size=(200, 100),
        color="white",
    )

    image.save(image_path)

    loader = ImageLoader()

    documents = loader.load(
        image_path
    )

    assert len(documents) == 1

    document = documents[0]

    assert document.is_image is True
    assert document.content == ""

    assert document.image_path == str(
        image_path
    )

    assert document.mime_type == (
        "image/png"
    )

    assert document.metadata[
        "width"
    ] == 200

    assert document.metadata[
        "height"
    ] == 100