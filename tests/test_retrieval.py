import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config.bootstrap import create_rag_pipeline


pipeline = create_rag_pipeline(
    "data/documents/AliChreimcv.pdf"
)


questions = [
    "What is the LinkedIn link?"
]


for question in questions:

    print()
    print("=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    response = pipeline.ask(
        question
    )

    results = response.search_results

    print(
        f"Retrieved: {len(results)} chunks"
    )

    for i, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            f"--- Result {i} ---"
        )

        print(
            "Score:",
            result.score,
        )

        print(
            "Source:",
            result.chunk.metadata.get(
                "source"
            ),
        )

        print(
            "Page:",
            result.chunk.metadata.get(
                "page"
            ),
        )

        print(
            "Content:"
        )

        print(
            result.chunk.content
        )