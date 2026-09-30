from config.bootstrap import create_rag_pipeline
from config.logging_config import configure_logging


def main():
    configure_logging()

    agent = create_rag_pipeline(
        "data/documents/AliChreimcv.pdf"
    )

    while True:
        question = input("\nQuestion: ")

        if question.lower() in {
            "exit",
            "quit",
        }:
            break

        try:
            answer = agent.ask(
                question
            )

            print()
            print("=== FINAL ANSWER ===")
            print(answer)

            print()
            print("=== PERFORMANCE ===")
            print(
                agent.last_performance
            )

        except Exception as error:
            print(
                f"\nError: {error}"
            )


if __name__ == "__main__":
    main()