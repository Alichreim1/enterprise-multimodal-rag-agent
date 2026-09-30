from config.bootstrap import create_rag_pipeline


DOCUMENT_PATH = "data/documents/AliChreimcv.pdf"


def test_real_rag_agent_success():
    agent = create_rag_pipeline(
        DOCUMENT_PATH
    )

    question = (
        "What is Ali's GitHub and what is 25 * 8?"
    )

    answer = agent.ask(question)

    print(
        "\n\n================ SUCCESS TEST ================\n"
    )

    print(answer)

    print(
        "\n================================================\n"
    )

    assert answer
    assert "github.com/Alichreim1" in answer
    assert "200" in answer


def test_real_rag_agent_unavailable():
    agent = create_rag_pipeline(
        DOCUMENT_PATH
    )

    question = (
        "What is Ali's favorite movie?"
    )

    answer = agent.ask(question)

    print(
        "\n\n================ UNAVAILABLE TEST ================\n"
    )

    print(answer)

    print(
        "\n====================================================\n"
    )

    assert answer


def test_real_rag_agent_statement():
    agent = create_rag_pipeline(
        DOCUMENT_PATH
    )

    question = (
        "I am building a RAG agent in Python."
    )

    answer = agent.ask(question)

    print(
        "\n\n================ STATEMENT TEST ================\n"
    )

    print(answer)

    print(
        "\n===================================================\n"
    )

    assert answer


def test_real_rag_agent_mixed():
    agent = create_rag_pipeline(
        DOCUMENT_PATH
    )

    question = (
        "What is Ali's GitHub and what is Ali's favorite movie?"
    )

    answer = agent.ask(question)

    print(
        "\n\n================ MIXED TEST ================\n"
    )

    print(answer)

    print(
        "\n===============================================\n"
    )

    assert answer