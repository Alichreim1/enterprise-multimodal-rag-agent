import time

from rag.agent.task_decomposer import TaskDecomposer
from rag.llms.ollama_llm import OllamaLLM


TEST_CASES = [

    # ============================================================
    # RETRIEVE
    # ============================================================

    {
        "question": "What is Ali's GitHub?",
        "expected_types": ["retrieve"],
        "expected_statements": [False],
    },
    {
        "question": "What is Ali's LinkedIn?",
        "expected_types": ["retrieve"],
        "expected_statements": [False],
    },
    {
        "question": "What is Ali's favorite programming language?",
        "expected_types": ["retrieve"],
        "expected_statements": [False],
    },
    {
        "question": "What is Ali's favorite movie?",
        "expected_types": ["retrieve"],
        "expected_statements": [False],
    },
    {
        "question": "Tell me what Python is.",
        "expected_types": ["retrieve"],
        "expected_statements": [False],
    },
    {
        "question": "Who is Ali?",
        "expected_types": ["retrieve"],
        "expected_statements": [False],
    },
    {
        "question": "What projects is Ali working on?",
        "expected_types": ["retrieve"],
        "expected_statements": [False],
    },
    {
        "question": "What programming language does Ali use?",
        "expected_types": ["retrieve"],
        "expected_statements": [False],
    },


    # ============================================================
    # TOOL
    # ============================================================

    {
        "question": "What is 25 * 8?",
        "expected_types": ["tool"],
        "expected_statements": [False],
    },
    {
        "question": "Calculate 3572 * 2.",
        "expected_types": ["tool"],
        "expected_statements": [False],
    },
    {
        "question": "What is 100 + 250?",
        "expected_types": ["tool"],
        "expected_statements": [False],
    },
    {
        "question": "Calculate 144 / 12.",
        "expected_types": ["tool"],
        "expected_statements": [False],
    },
    {
        "question": "What is 15 squared?",
        "expected_types": ["tool"],
        "expected_statements": [False],
    },
    {
        "question": "Calculate 50 * 20 + 10.",
        "expected_types": ["tool"],
        "expected_statements": [False],
    },
    {
        "question": "How much is 999 - 456?",
        "expected_types": ["tool"],
        "expected_statements": [False],
    },


    # ============================================================
    # STATEMENTS
    # ============================================================

    {
        "question": "I prefer Python.",
        "expected_types": ["statement"],
        "expected_statements": [True],
    },
    {
        "question": "I like basketball.",
        "expected_types": ["statement"],
        "expected_statements": [True],
    },
    {
        "question": "I am building a RAG agent in Python.",
        "expected_types": ["statement"],
        "expected_statements": [True],
    },
    {
        "question": "My favorite programming language is Python.",
        "expected_types": ["statement"],
        "expected_statements": [True],
    },
    {
        "question": "I want to build a Jarvis-like AI assistant.",
        "expected_types": ["statement"],
        "expected_statements": [True],
    },


    # ============================================================
    # MULTI TASK - RETRIEVE + TOOL
    # ============================================================

    {
        "question": "What is Ali's GitHub and what is 25 * 8?",
        "expected_types": ["retrieve", "tool"],
        "expected_statements": [False, False],
    },
    {
        "question": "What is Ali's LinkedIn and what is 100 + 50?",
        "expected_types": ["retrieve", "tool"],
        "expected_statements": [False, False],
    },
    {
        "question": "What is Ali's favorite programming language and what is 20 * 5?",
        "expected_types": ["retrieve", "tool"],
        "expected_statements": [False, False],
    },
    {
        "question": "What is 3572 * 2 and what is Ali's GitHub?",
        "expected_types": ["tool", "retrieve"],
        "expected_statements": [False, False],
    },


    # ============================================================
    # MULTI RETRIEVE
    # ============================================================

    {
        "question": "What is Ali's GitHub and what is his LinkedIn?",
        "expected_types": ["retrieve", "retrieve"],
        "expected_statements": [False, False],
    },
    {
        "question": "What is Ali's GitHub, LinkedIn, and favorite programming language?",
        "expected_types": ["retrieve", "retrieve", "retrieve"],
        "expected_statements": [False, False, False],
    },


    # ============================================================
    # MULTI TOOL
    # ============================================================

    {
        "question": "What is 25 * 8 and what is 100 / 5?",
        "expected_types": ["tool", "tool"],
        "expected_statements": [False, False],
    },
    {
        "question": "Calculate 20 + 30 and then calculate 50 * 2.",
        "expected_types": ["tool", "tool"],
        "expected_statements": [False, False],
    },


    # ============================================================
    # MIXED / LONGER
    # ============================================================

    {
        "question": (
            "What is Ali's GitHub, what is his LinkedIn, "
            "and what is 25 * 8?"
        ),
        "expected_types": ["retrieve", "retrieve", "tool"],
        "expected_statements": [False, False, False],
    },
    {
        "question": (
            "What is 3572 * 2, what is Ali's GitHub, "
            "and what is Ali's LinkedIn?"
        ),
        "expected_types": ["tool", "retrieve", "retrieve"],
        "expected_statements": [False, False, False],
    },
    {
        "question": (
            "I prefer Python and I like basketball."
        ),
        "expected_types": ["statement", "statement"],
        "expected_statements": [True, True],
    },
    {
        "question": (
            "I am building a RAG agent in Python and what is Ali's GitHub?"
        ),
        "expected_types": ["statement", "retrieve"],
        "expected_statements": [True, False],
    },
    {
        "question": (
            "I like basketball. What is Ali's GitHub? "
            "And what is 10 * 10?"
        ),
        "expected_types": ["statement", "retrieve", "tool"],
        "expected_statements": [True, False, False],
    },
]


def evaluate_model(model_name: str):
    print("\n")
    print("=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    llm = OllamaLLM(model_name)
    decomposer = TaskDecomposer(llm)

    total = len(TEST_CASES)

    valid_json = 0
    correct_count = 0
    correct_types = 0
    correct_statements = 0

    total_latency = 0

    results = []

    for index, case in enumerate(TEST_CASES, start=1):

        question = case["question"]
        expected_types = case["expected_types"]
        expected_statements = case["expected_statements"]

        print("\n" + "-" * 80)
        print(f"TEST {index}/{total}")
        print(f"QUESTION: {question}")

        start = time.perf_counter()

        try:
            tasks = decomposer.decompose(question)

            latency = time.perf_counter() - start
            total_latency += latency

            valid_json += 1

            actual_types = [task.task_type for task in tasks]
            actual_statements = [task.is_statement for task in tasks]

            count_ok = len(tasks) == len(expected_types)
            types_ok = actual_types == expected_types
            statements_ok = actual_statements == expected_statements

            if count_ok:
                correct_count += 1

            if types_ok:
                correct_types += 1

            if statements_ok:
                correct_statements += 1

            passed = (
                count_ok
                and types_ok
                and statements_ok
            )

            results.append({
                "index": index,
                "question": question,
                "passed": passed,
                "actual_types": actual_types,
                "expected_types": expected_types,
                "actual_statements": actual_statements,
                "expected_statements": expected_statements,
                "latency": latency,
            })

            print(f"EXPECTED TYPES:      {expected_types}")
            print(f"ACTUAL TYPES:        {actual_types}")

            print(f"EXPECTED STATEMENTS: {expected_statements}")
            print(f"ACTUAL STATEMENTS:   {actual_statements}")

            print(f"COUNT OK:            {count_ok}")
            print(f"TYPES OK:            {types_ok}")
            print(f"STATEMENTS OK:       {statements_ok}")
            print(f"LATENCY:             {latency:.2f}s")

            print(
                "RESULT:              "
                + ("PASS" if passed else "FAIL")
            )

        except Exception as exc:

            latency = time.perf_counter() - start
            total_latency += latency

            results.append({
                "index": index,
                "question": question,
                "passed": False,
                "actual_types": None,
                "expected_types": expected_types,
                "actual_statements": None,
                "expected_statements": expected_statements,
                "latency": latency,
                "error": str(exc),
            })

            print(f"ERROR: {exc}")
            print("RESULT: FAIL")

    print("\n")
    print("=" * 80)
    print(f"FINAL RESULTS — {model_name}")
    print("=" * 80)

    print(f"Total tests:             {total}")
    print(f"Valid JSON:              {valid_json}/{total}")
    print(f"Correct task count:      {correct_count}/{total}")
    print(f"Correct task types:      {correct_types}/{total}")
    print(f"Correct statements:      {correct_statements}/{total}")

    if total > 0:
        print(
            f"JSON validity:           "
            f"{valid_json / total * 100:.1f}%"
        )

        print(
            f"Task count accuracy:     "
            f"{correct_count / total * 100:.1f}%"
        )

        print(
            f"Task type accuracy:      "
            f"{correct_types / total * 100:.1f}%"
        )

        print(
            f"Statement accuracy:      "
            f"{correct_statements / total * 100:.1f}%"
        )

    if valid_json > 0:
        print(
            f"Average latency:         "
            f"{total_latency / valid_json:.2f}s"
        )

    print("\nFAILED TESTS")
    print("-" * 80)

    failed = [result for result in results if not result["passed"]]

    if not failed:
        print("None")
    else:
        for result in failed:
            print(
                f"\nTest {result['index']}: "
                f"{result['question']}"
            )

            print(
                f"Expected: {result['expected_types']}"
            )

            print(
                f"Actual:   {result['actual_types']}"
            )

            if "error" in result:
                print(f"Error: {result['error']}")

    return results


if __name__ == "__main__":

    evaluate_model("qwen2.5:3b")
    evaluate_model("qwen3:8b")