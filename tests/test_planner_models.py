import time

from rag.agent.planner import Planner
from rag.llms.ollama_llm import OllamaLLM


TEST_CASES = [

    # ============================================================
    # RETRIEVE
    # ============================================================

    {
        "task": "What is Ali's GitHub?",
        "expected_action": "retrieve",
    },
    {
        "task": "What is Ali's LinkedIn?",
        "expected_action": "retrieve",
    },
    {
        "task": "What is Ali's favorite programming language?",
        "expected_action": "retrieve",
    },
    {
        "task": "What is Ali's favorite movie?",
        "expected_action": "retrieve",
    },
    {
        "task": "Tell me what Python is.",
        "expected_action": "retrieve",
    },
    {
        "task": "Who is Ali?",
        "expected_action": "retrieve",
    },
    {
        "task": "What projects is Ali working on?",
        "expected_action": "retrieve",
    },
    {
        "task": "What programming language does Ali use?",
        "expected_action": "retrieve",
    },


    # ============================================================
    # TOOL
    # ============================================================

    {
        "task": "What is 25 * 8?",
        "expected_action": "tool",
    },
    {
        "task": "Calculate 3572 * 2.",
        "expected_action": "tool",
    },
    {
        "task": "What is 100 + 250?",
        "expected_action": "tool",
    },
    {
        "task": "Calculate 144 / 12.",
        "expected_action": "tool",
    },
    {
        "task": "What is 15 squared?",
        "expected_action": "tool",
    },
    {
        "task": "Calculate 50 * 20 + 10.",
        "expected_action": "tool",
    },
    {
        "task": "How much is 999 - 456?",
        "expected_action": "tool",
    },


    # ============================================================
    # FINISH
    # ============================================================

    {
        "task": "I prefer Python.",
        "expected_action": "finish",
    },
    {
        "task": "I like basketball.",
        "expected_action": "finish",
    },
    {
        "task": "I am building a RAG agent in Python.",
        "expected_action": "finish",
    },
    {
        "task": "My favorite programming language is Python.",
        "expected_action": "finish",
    },
    {
        "task": "I want to build a Jarvis-like AI assistant.",
        "expected_action": "finish",
    },


    # ============================================================
    # MORE RETRIEVE
    # ============================================================

    {
        "task": "Where can I find Ali's GitHub profile?",
        "expected_action": "retrieve",
    },
    {
        "task": "Can you tell me Ali's LinkedIn profile?",
        "expected_action": "retrieve",
    },
    {
        "task": "What technologies does Ali know?",
        "expected_action": "retrieve",
    },


    # ============================================================
    # MORE TOOL
    # ============================================================

    {
        "task": "Multiply 123 by 456.",
        "expected_action": "tool",
    },
    {
        "task": "What is 500 divided by 25?",
        "expected_action": "tool",
    },
    {
        "task": "Calculate 75 + 125 - 50.",
        "expected_action": "tool",
    },


    # ============================================================
    # STATEMENT / FINISH
    # ============================================================

    {
        "task": "I enjoy reading books.",
        "expected_action": "finish",
    },
    {
        "task": "I am learning artificial intelligence.",
        "expected_action": "finish",
    },
    {
        "task": "I want to create my own AI assistant.",
        "expected_action": "finish",
    },
]


def evaluate_model(model_name: str):

    print("\n")
    print("=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    llm = OllamaLLM(model_name)
    planner = Planner(llm)

    total = len(TEST_CASES)

    valid_json = 0
    correct_actions = 0

    total_latency = 0

    results = []

    for index, case in enumerate(TEST_CASES, start=1):

        task = case["task"]
        expected_action = case["expected_action"]

        print("\n" + "-" * 80)
        print(f"TEST {index}/{total}")
        print(f"TASK: {task}")

        start = time.perf_counter()

        try:

            decision = planner.plan(task)

            latency = time.perf_counter() - start
            total_latency += latency

            valid_json += 1

            actual_action = decision.action

            action_ok = actual_action == expected_action

            if action_ok:
                correct_actions += 1

            results.append({
                "index": index,
                "task": task,
                "expected": expected_action,
                "actual": actual_action,
                "passed": action_ok,
                "latency": latency,
            })

            print(
                f"EXPECTED ACTION: {expected_action}"
            )

            print(
                f"ACTUAL ACTION:   {actual_action}"
            )

            print(
                f"REASON:          {decision.reason}"
            )

            print(
                f"ACTION OK:       {action_ok}"
            )

            print(
                f"LATENCY:         {latency:.2f}s"
            )

            print(
                "RESULT:          "
                + ("PASS" if action_ok else "FAIL")
            )

        except Exception as exc:

            latency = time.perf_counter() - start
            total_latency += latency

            results.append({
                "index": index,
                "task": task,
                "expected": expected_action,
                "actual": None,
                "passed": False,
                "latency": latency,
                "error": str(exc),
            })

            print(f"ERROR: {exc}")
            print("RESULT: FAIL")

    # ============================================================
    # FINAL RESULTS
    # ============================================================

    print("\n")
    print("=" * 80)
    print(f"FINAL RESULTS — {model_name}")
    print("=" * 80)

    print(f"Total tests:             {total}")
    print(f"Valid JSON:              {valid_json}/{total}")
    print(f"Correct actions:         {correct_actions}/{total}")

    if total > 0:

        print(
            f"JSON validity:           "
            f"{valid_json / total * 100:.1f}%"
        )

        print(
            f"Action accuracy:         "
            f"{correct_actions / total * 100:.1f}%"
        )

    if valid_json > 0:

        print(
            f"Average latency:         "
            f"{total_latency / valid_json:.2f}s"
        )

    # ============================================================
    # BREAKDOWN
    # ============================================================

    print("\nACTION BREAKDOWN")
    print("-" * 80)

    categories = ["retrieve", "tool", "finish"]

    for category in categories:

        category_results = [
            result
            for result in results
            if result["expected"] == category
        ]

        category_total = len(category_results)

        category_correct = sum(
            1
            for result in category_results
            if result["passed"]
        )

        if category_total > 0:

            accuracy = (
                category_correct
                / category_total
                * 100
            )

            print(
                f"{category:10} "
                f"{category_correct}/{category_total} "
                f"({accuracy:.1f}%)"
            )

    # ============================================================
    # FAILED TESTS
    # ============================================================

    print("\nFAILED TESTS")
    print("-" * 80)

    failed = [
        result
        for result in results
        if not result["passed"]
    ]

    if not failed:

        print("None")

    else:

        for result in failed:

            print(
                f"\nTest {result['index']}:"
            )

            print(
                f"Task:     {result['task']}"
            )

            print(
                f"Expected: {result['expected']}"
            )

            print(
                f"Actual:   {result['actual']}"
            )

            if "error" in result:
                print(
                    f"Error:    {result['error']}"
                )

    return results


if __name__ == "__main__":

    evaluate_model("qwen2.5:3b")

    evaluate_model("qwen3:8b")