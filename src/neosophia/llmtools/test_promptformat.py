"""
Tests for prompt formatting
"""


from neosophia.llmtools import openaiapi as oai

from neosophia.llmtools import promptformat as pf


EXAMPLE_MESSAGES = [
    oai.Message(role='system', content='You are a chat model.'),
    oai.Message(role='user', content='Hello'),
    oai.Message(role='assistant', content='world'),
    oai.Message(role='user', content='What now?'),
]


def test_messages_to_llama2_prompt():
    """test llama2 format"""

    res = pf.messages_to_llama2_prompt(EXAMPLE_MESSAGES)

    expected = (
        '[INST] <<SYS>>\nYou are a chat model.\n<</SYS>>\n\n' +
        'Hello [/INST]\n\n' +
        'world\n\n' +
        '[INST] What now? [/INST]'
    )

    assert res == expected


def test_messages_to_alpaca_prompt():
    """test alpaca format"""

    res = pf.messages_to_alpaca_prompt(EXAMPLE_MESSAGES)

    expected = (
        'You are a chat model.\n\n' +
        '### Instruction:\nHello\n\n' +
        '### Response:\nworld\n\n' +
        '### Instruction:\nWhat now?\n\n' +
        '### Response:'
    )

    assert res == expected


def test_messages_to_chatml_prompt():
    """test alpaca format"""

    res = pf.messages_to_chatml_prompt(EXAMPLE_MESSAGES)

    expected = (
        '<|im_start|>system\nYou are a chat model.<|im_end|>\n\n' +
        '<|im_start|>user\nHello<|im_end|>\n\n' +
        '<|im_start|>assistant\nworld<|im_end|>\n\n' +
        '<|im_start|>user\nWhat now?<|im_end|>\n\n' +
        '<|im_start|>assistant\n'
    )

    assert res == expected


def test_messages_to_phind_prompt():
    """test alpaca format"""

    res = pf.messages_to_phind_prompt(EXAMPLE_MESSAGES)

    expected = (
        '### System Prompt:\nYou are a chat model.\n\n' +
        '### User Message:\nHello\n\n' +
        '### Assistant:\nworld\n\n' +
        '### User Message:\nWhat now?\n\n' +
        '### Assistant:'
    )

    assert res == expected


def test_messages_to_vicuna_prompt():
    """test vicuna format"""

    res = pf.messages_to_vicuna_prompt(EXAMPLE_MESSAGES)

    expected = (
        'You are a chat model.\n\n' +
        'USER:\nHello\n\n' +
        'ASSISTANT:\nworld\n\n' +
        'USER:\nWhat now?\n\n' +
        'ASSISTANT:\n'
    )

    assert res == expected
