""" Module for holding data classes """
from typing import Any, Callable, Optional
from dataclasses import dataclass

from neosophia.llmtools import openaiapi as oaiapi


@dataclass
class Variable:
    """ Class to hold a Variable that the agent can use """
    name: str
    value: Any
    description: str
    visible: bool = True

    # Dynamic is True if the Variable was created by the Agent
    dynamic: bool = False

    def to_string(self):
        output = f'\nName: {self.name}\n'
        output += f'Value:\n{self.value}\n'
        output += f'Description: {self.description}\n'
        return output


@dataclass
class Tool:
    """ Class to hold a Tool which is a Python function that the agent can use """
    name: str
    function_str: str
    description: str
    call: Callable

    def to_string(self):
        output = f'\nTool Name: {self.name}\n'
        output += f'Description: {self.description}\n'
        return output


class Colors:
    """ Colors for printing """
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[94m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    ENDC = '\033[0m'


@dataclass
class GPTModelInfo:
    """ Class to hold information about a GPT model """
    name: str
    input_token_cost: float
    output_token_cost: float
    max_tokens: int

# https://openai.com/pricing#language-models
GPT_MODELS = {
    info.name: info
    for info in [
        GPTModelInfo(
            name='gpt-3.5-turbo-0301',
            input_token_cost=0.0015 / 1000,
            output_token_cost=0.002 / 1000,
            max_tokens=4096,
        ),
        GPTModelInfo(
            name='gpt-3.5-turbo-0613',
            input_token_cost=0.0015 / 1000,
            output_token_cost=0.002 / 1000,
            max_tokens=4096,
        ),
        GPTModelInfo(
            name='gpt-3.5-turbo-16k-0613',
            input_token_cost=0.003 / 1000,
            output_token_cost=0.004 / 1000,
            max_tokens=16384,
        ),
        GPTModelInfo(
            name='gpt-4',
            input_token_cost=0.03 / 1000,
            output_token_cost=0.06 / 1000,
            max_tokens=8192,
        ),
        GPTModelInfo(
            name='gpt-4-0314',
            input_token_cost=0.03 / 1000,
            output_token_cost=0.06 / 1000,
            max_tokens=8192,
        ),
        GPTModelInfo(
            name='gpt-4-0613',
            input_token_cost=0.03 / 1000,
            output_token_cost=0.06 / 1000,
            max_tokens=8192,
        )
    ]
}

