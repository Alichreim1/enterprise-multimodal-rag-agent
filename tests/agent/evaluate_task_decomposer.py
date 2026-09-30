import time

from rag.agent.task_decomposer import TaskDecomposer
from rag.llms.ollama_llm import OllamaLLM


TEST_CASES = [
    {
        "question": "What is Ali's GitHub?",
        "expected_types": ["retrieve"],
    },
    {
        "question": "What is Ali's LinkedIn?",
        "expected_types": ["retrieve"],
    },
    {
        "question": "What is 25 * 8?",
        "expected_types": ["tool"],
    },
    {
        "question": "What is Ali's GitHub and what is 25 * 8?",
        "expected_types": ["retrieve", "tool"],
    },
    {
        "question": "What is Ali's GitHub and what is his LinkedIn?",
        "expected_types": ["retrieve", "retrieve"],
    },
    {
        "question": "What is Ali's favorite programming language?",
        "expected_types": ["retrieve"],
    },
    {
        "question": "Tell me what Python is.",
        "expected_types": ["retrieve"],
    },
    {
        "question": "I prefer Python.",
        "expected_types": ["statement"],
    },
    {
        "question": "I like basketball.",
        "expected_types": ["statement"],
    },
    {
        "question": "I am building a RAG agent in Python.",
        "expected_types": ["statement"],
    },
    {
        "question": (
            "What is Ali's GitHub, "
            "what is his LinkedIn, and what is 25 * 8?"
        ),
        "expected_types": ["retrieve", "retrieve", "tool"],
    },
    {
        "question": "What is 3572 * 2 and what is Ali's GitHub?",
        "expected_types": ["tool", "retrieve"],
    },
]


def evaluate_model(model_name: str, llm: OllamaLLM) -> None:
    decomposer = TaskDecomposer(llm)

    total = len(TEST_CASES)
    valid_json = 0
    correct_count = 0
    correct_types = 0
    total_time = 0.0

    print()
    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    for index, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected_types = test["expected_types"]

        start = time.perf_counter()

        try:
            tasks = decomposer.decompose(question)
            elapsed = time.perf_counter() - start

            total_time += elapsed
            valid_json += 1

            actual_types = [task.task_type for task in tasks]

            count_ok = len(tasks) == len(expected_types)
            types_ok = actual_types == expected_types

            if count_ok:
                correct_count += 1

            if types_ok:
                correct_types += 1

            print()
            print(f"Test {index}")
            print(f"Question: {question}")
            print(f"Expected: {expected_types}")
            print(f"Actual:   {actual_types}")
            print(f"Count:    {'PASS' if count_ok else 'FAIL'}")
            print(f"Types:    {'PASS' if types_ok else 'FAIL'}")
            print(f"Time:     {elapsed:.2f}s")

        except Exception as exc:
            print()
            print(f"Test {index}")
            print(f"Question: {question}")
            print(f"ERROR: {exc}")

    print()
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)

    print(f"Valid output:     {valid_json}/{total}")
    print(f"Correct count:    {correct_count}/{total}")
    print(f"Correct types:    {correct_types}/{total}")

    if total_time > 0:
        average_latency = total_time / valid_json if valid_json else 0
        print(f"Average latency:  {average_latency:.2f}s")

    print("=" * 70)


def main() -> None:
    models = [
        ("qwen2.5:3b", OllamaLLM("qwen2.5:3b")),
        ("qwen3:8b", OllamaLLM("qwen3:8b")),
    ]

    for model_name, llm in models:
        evaluate_model(model_name, llm)


if __name__ == "__main__":
    main()
