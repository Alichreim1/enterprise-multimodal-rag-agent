import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.bootstrap import create_rag_pipeline


def test_rag():

    rag = create_rag_pipeline(
        "data/documents/NovaAI_Employee_Handbook.pdf"
    )

    questions = [
        "What is NovaAI?",
        "Who is Ali Chreim?",
        "What services does NovaAI provide?",
    ]

    for question in questions:

        print("\n")
        print("=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        response = rag.ask(question)

        print("\n--- RETRIEVED RESULTS ---")

        for i, result in enumerate(
            response.search_results,
            start=1,
        ):

            print(
                f"\nResult {i}"
            )

            print(
                f"Score: {result.score:.4f}"
            )

            print(
                f"Source: "
                f"{result.chunk.metadata.get('source')}"
            )

            print(
                f"Page: "
                f"{result.chunk.metadata.get('page')}"
            )

            print(
                f"Content:\n"
                f"{result.chunk.content}"
            )

        print("\n--- PROMPT ---")
        print(response.prompt)

        print("\n--- ANSWER ---")
        print(response.answer)