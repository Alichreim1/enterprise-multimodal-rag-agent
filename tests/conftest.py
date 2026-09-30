import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from rag.models.document import Document


def sample_document():

    return Document(

        content=(
            "Python is an amazing language. "
            * 200
        ),

        metadata={
            "source": "sample.pdf",
            "page": 1,
            "file_type": "pdf",
        },

    )