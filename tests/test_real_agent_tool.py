from tests.test_real_agent import create_test_agent


def test_real_agent_retrieve_and_tool():

    agent = create_test_agent()

    answer = agent.ask(
        "What is Ali's experience and what is 25 * 4?"
    )

    print("\nREAL AGENT ANSWER:")
    print(answer)

    assert answer