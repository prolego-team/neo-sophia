""" Module for holding data classes """
from typing import Any, Callable, Optional
from dataclasses import dataclass

from neosophia.llmtools import openaiapi as oaiapi
from neosophia.agents.system_prompts import VARIABLE_SUMMARY_PROMPT


@dataclass
class Resource:
    """ Class to hold a data Resource that the agent can access """
    name: str
    path: str
    description: str
    visible: bool = True

    def __str__(self):

        output = f'\n{Colors.BLUE}Resource Name: {Colors.ENDC}{self.name}\n'
        output += f'Path: {self.path}\n'
        output += f'Decription: {self.description}'
        output += f'Visible: {self.visible}'
        return output


@dataclass
class Variable:
    """ Class to hold a Variable that the agent can use """
    name: str
    value: Any
    description: str
    summary: Optional[str] = None
    visible: bool = True

    def to_string(self):
        output = f'\n{Colors.BLUE}Name: {Colors.ENDC}{self.name}\n'
        output += f'{Colors.BLUE}Value:\n{Colors.ENDC}{self.value}\n'
        output += f'{Colors.BLUE}Description: {Colors.ENDC}{self.description}\n'
        return output

    def get_summary(self, model_name):
        """ Calls a LLM to obtain a summary about the variable """
        prompt = VARIABLE_SUMMARY_PROMPT + self.to_string()
        self.summary = oaiapi.chat_completion(prompt=prompt, model=model_name)
        return self.summary

    def __str__(self):
        output = self.to_string()
        if self.summary is not None:
            output += f'{Colors.BLUE}Summary: {Colors.ENDC}{self.summary}\n'
        return output


@dataclass
class Tool:
    """ Class to hold a Tool which is a Python function that the agent can use """
    name: str
    function_str: str
    description: str
    call: Callable

    def __str__(self):
        output = f'\n{Colors.BLUE}Tool Name: {Colors.ENDC}{self.name}\n'
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

