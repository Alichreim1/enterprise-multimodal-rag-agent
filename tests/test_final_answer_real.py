from rag.agent.final_answer import FinalAnswerGenerator
from rag.llms.ollama_llm import OllamaLLM
from rag.models.task import AgentTask


def test_real_model_final_answer():

    llm = OllamaLLM(
        model="qwen2.5:3b"
    )

    generator = FinalAnswerGenerator(llm)

    tasks = [
        AgentTask(
            id=0,
            description="What is Ali's GitHub?",
            task_type="retrieve",
            completed=True,
            result=(
                "Ali's GitHub is "
                "https://github.com/Alichreim1/."
            ),
        ),
        AgentTask(
            id=1,
            description="What is 25 * 8?",
            task_type="tool",
            completed=True,
            result="25 * 8 = 200",
        ),
    ]

    answer = generator.generate(
        question=(
            "What is Ali's GitHub and what is 25 * 8?"
        ),
        tasks=tasks,
    )

    print("\n================ FINAL ANSWER ================\n")
    print(answer)
    print("\n===============================================\n")

    assert answer
    assert "github.com/Alichreim1" in answer
    assert "200" in answer


def test_real_model_does_not_invent_information():

    llm = OllamaLLM(
        model="qwen2.5:3b"
    )

    generator = FinalAnswerGenerator(llm)

    tasks = [
        AgentTask(
            id=0,
            description="What is Ali's favorite programming language?",
            task_type="retrieve",
            completed=False,
            result=(
                "The requested information could not be "
                "found in the available documents."
            ),
        ),
    ]

    answer = generator.generate(
        question=(
            "What is Ali's favorite programming language?"
        ),
        tasks=tasks,
    )

    print("\n================ FINAL ANSWER ================\n")
    print(answer)
    print("\n===============================================\n")

    assert answer

    # The model should not invent a programming language.
    forbidden = [
        "Python",
        "Java",
        "JavaScript",
        "C++",
        "C#",
        "Go",
        "Rust",
    ]

    for language in forbidden:
        assert language.lower() not in answer.lower()


def test_real_model_preserves_only_provided_information():

    llm = OllamaLLM(
        model="qwen2.5:3b"
    )

    generator = FinalAnswerGenerator(llm)

    tasks = [
        AgentTask(
            id=0,
            description="What is Ali's GitHub?",
            task_type="retrieve",
            completed=True,
            result=(
                "Ali's GitHub is "
                "https://github.com/Alichreim1/."
            ),
        ),
        AgentTask(
            id=1,
            description="What is Ali's LinkedIn?",
            task_type="retrieve",
            completed=True,
            result=(
                "Ali's LinkedIn is "
                "https://www.linkedin.com/in/ali-chreim/"
                "397626234/."
            ),
        ),
    ]

    answer = generator.generate(
        question=(
            "What are Ali's GitHub and LinkedIn?"
        ),
        tasks=tasks,
    )

    print("\n================ FINAL ANSWER ================\n")
    print(answer)
    print("\n===============================================\n")

    assert answer

    assert "github.com/Alichreim1" in answer
    assert "linkedin.com/in/ali-chreim" in answer

    # Information that was not provided should not appear.
    assert "Twitter" not in answer
    assert "Facebook" not in answer
    assert "Instagram" not in answer