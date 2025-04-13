"""Integration test for the agent stream functionality."""  # noqa: INP001

from ibis_crew_ai.agent import agent_workflow


def test_agent_stream() -> None:
    """Integration test for the agent stream functionality.

    Tests that the agent returns valid streaming responses.
    """
    input_dict = {'messages': [{'type': 'human',
                                'content': 'Hi'},
                               {'type': 'ai',
                                'content': 'Hi there!'},
                               {'type': 'human',
                                'content': 'Write a fibonacci function in python'}]}
    agent = agent_workflow()
    events = [message for message, _ in agent.stream(input_dict, stream_mode='messages')]
    # Verify we get a reasonable number of messages
    assert len(events) > 0, 'Expected at least one message'

    # First message should be an AI message
    assert events[0].type == 'AIMessageChunk'

    # At least one message should have content
    has_content = False
    for event in events:
        if hasattr(event, 'content') and event.content:
            has_content = True
            break
    assert has_content, 'Expected at least one message with content'
