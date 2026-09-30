from rag.agent.reflection import Reflection
from rag.llms.ollama_llm import OllamaLLM


def test_reflection_real():
    llm = OllamaLLM(
        model="qwen3:8b"
    )

    reflection = Reflection(llm)

    test_cases = [
        {
            "name": "correct retrieval",
            "task": "What is Ali's GitHub?",
            "result": (
                "Ali's GitHub is "
                "https://github.com/Alichreim1/."
            ),
            "expected": True,
        },
        {
            "name": "information unavailable",
            "task": "What is Ali's favorite programming language?",
            "result": (
                "The requested information could not "
                "be found in the available documents."
            ),
            "expected": False,
        },
        {
            "name": "irrelevant result",
            "task": "What is Ali's LinkedIn?",
            "result": (
                "Ali's GitHub is "
                "https://github.com/Alichreim1/."
            ),
            "expected": False,
        },
        {
            "name": "partial result",
            "task": "What is Ali's GitHub and LinkedIn?",
            "result": (
                "Ali's GitHub is "
                "https://github.com/Alichreim1/."
            ),
            "expected": False,
        },
        {
            "name": "calculator correct",
            "task": "What is 25 * 8?",
            "result": "25 * 8 = 200",
            "expected": True,
        },
        {
            "name": "empty result",
            "task": "What is Ali's GitHub?",
            "result": "",
            "expected": False,
        },
        {
            "name": "irrelevant calculator result",
            "task": "What is Ali's GitHub?",
            "result": "25 * 8 = 200",
            "expected": False,
        },
    ]

    for case in test_cases:
        decision = reflection.reflect(
            task=case["task"],
            result=case["result"],
        )

        print("\n================================")
        print("TEST:", case["name"])
        print("TASK:", case["task"])
        print("RESULT:", case["result"])
        print("COMPLETED:", decision.completed)
        print("REASON:", decision.reason)
        print("EXPECTED:", case["expected"])
        print("================================")

        assert decision.completed == case["expected"]
