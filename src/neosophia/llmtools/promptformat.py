"""
Prompt formatting for different LLMs.

Documentation on formatting is not the best, so there may be issues
with these.
"""

from typing import List

from neosophia.llmtools import openaiapi as openai


STOP_LLAMA = ['[INST', '[Inst']
STOP_ALPACA = ['### ']
STOP_CHATML = ['<|im_end|>', '<|im_start|>']
STOP_PHIND = ['### ']
STOP_VICUNA = ['USER:']


def messages_to_llama2_prompt(messages: List[openai.Message]) -> str:
    """
    Convert a list of messages representing a chat conversation
    to a llama2 prompt.
    """

    # Reference on how it's officially done:
    # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212
    # What I've implemented isn't quite the same, I think there might
    # be some additional beginning and end of sentence tokensization
    # going on in the original method.

    assert len(messages) > 0, 'need at least 1 message'

    if len(messages) > 1 and messages[0].role == 'system':

        assert messages[0].role == 'system', 'wrong role for first message'
        assert messages[1].role == 'user', 'wrong role for second message'

        # special case for putting sys tags inside INST
        messages_text = [
            f'[INST] <<SYS>>\n{messages[0].content}\n<</SYS>>\n\n{messages[1].content} [/INST]'
        ]
        idx_start = 2
    else:
        messages_text = []
        idx_start = 0

    for message in messages[idx_start:]:
        content = message.content.strip()
        if message.role != 'assistant':
            content = f'[INST] {content} [/INST]'
        messages_text.append(content)

    res = '\n\n'.join(messages_text)

    return res


def messages_to_alpaca_prompt(messages: List[openai.Message]) -> str:
    """Convert a series of messages to Alpaca format."""

    assert len(messages) > 0, 'not enough messages'

    if messages[0].role == 'system':
        messages_text = [messages[0].content + '\n']
        idx_start = 1
    else:
        messages_text = []
        idx_start = 0

    for message in messages[idx_start:]:
        content = message.content.strip()
        if message.role == 'assistant':
            tag = 'Response'
        else:
            tag = 'Instruction'
        messages_text.append(f'### {tag}:\n{content}\n')

    res = '\n'.join(messages_text) + '\n### Response:'

    # TODO: should we add another newline at the end???

    return res


def messages_to_phind_prompt(messages: List[openai.Message]) -> str:
    """Convert a series of messages to Phind format."""

    assert len(messages) > 0, 'not enough messages'
    messages_text = []

    for message in messages:
        content = message.content.strip()
        if message.role == 'system':
            tag = 'System Prompt'
        elif message.role == 'assistant':
            tag = 'Assistant'
        else:
            tag = 'User Message'
        messages_text.append(f'### {tag}:\n{content}\n')

    res = '\n'.join(messages_text) + '\n### Assistant:'

    # TODO: additional newlines?

    return res


def messages_to_chatml_prompt(messages: List[openai.Message]) -> str:
    """Convert a series of messages to chatml format."""

    messages_text = []
    for message in messages:
        content = message.content.strip()
        role = message.role
        if role == 'function':
            role = 'user'
        messages_text.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')

    res = '\n'.join(messages_text) + '\n<|im_start|>assistant\n'

    # TODO: additional newline?

    return res


def messages_to_vicuna_prompt(messages: List[openai.Message]) -> str:
    """Not sure about this. None of the examples have newlines."""

    assert len(messages) > 0, 'not enough messages'

    if messages[0].role == 'system':
        messages_text = [messages[0].content + '\n']
        idx_start = 1
    else:
        messages_text = []
        idx_start = 0

    for message in messages[idx_start:]:
        content = message.content.strip()
        if message.role == 'assistant':
            tag = 'ASSISTANT'
        else:
            tag = 'USER'
        messages_text.append(f'{tag}:\n{content}\n')

    res = '\n'.join(messages_text) + '\nASSISTANT:\n'

    return res
