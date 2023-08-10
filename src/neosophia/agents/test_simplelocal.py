"""
Tests for simple local agent related funcionality.
"""


from neosophia.agents import simplelocal
from neosophia.llmtools import openaiapi as oai


def test_messages_to_llama2_prompt():
    """test converting message objects to a llama2 prompt"""

    res = simplelocal.messages_to_llama2_prompt([
        oai.Message(role='system', content='You are a chat model.'),
        oai.Message(role='user', content='Hello'),
        oai.Message(role='assistant', content='world'),
        oai.Message(role='user', content='What now?'),
    ])

    print(res)

    assert (
        res ==
        (
            '[INST] <<SYS>>\nYou are a chat model.\n<</SYS>>\n\n' +
            'Hello [/INST]\n\n' +
            'world\n\n' +
            '[INST] What now? [/INST]'
        )
    )
