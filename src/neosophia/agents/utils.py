"""
"""
import ast
import types

from typing import Callable, Dict, List, Tuple

import tiktoken
import astunparse

from neosophia.llmtools import openaiapi as oaiapi
from neosophia.agents.system_prompts import FUNCTION_GPT_PROMPT


def _build_function_dict_from_modules(
        modules: List[types.ModuleType]) -> List[str]:
    """
    Takes a list of python modules as input and builds a dictionary containing
    the function name as the key and the function itself as a string as the
    value
    """

    function_dict = {}
    for module in modules:
        with open (module.__file__, 'r') as f:
            function_text = ''.join(f.readlines())
        for node in ast.walk(ast.parse(function_text)):
            if isinstance(node, ast.FunctionDef):
                function_dict[node.name] = astunparse.unparse(node)

    return function_dict


def build_function_dict_from_modules(
        modules: List[types.ModuleType]) -> Dict[str, Tuple[Callable, str]]:
    """
    Takes a list of python modules as input and builds a dictionary containing
    the function name as the key and a tuple containing the callable function
    and its string representation as the value.
    """

    function_dict = {}
    for module in modules:
        with open(module.__file__, 'r') as f:
            function_text = ''.join(f.readlines())
        for node in ast.walk(ast.parse(function_text)):
            if isinstance(node, ast.FunctionDef):
                callable_function = getattr(module, node.name)
                function_dict[node.name] = (
                    callable_function, astunparse.unparse(node))

    return function_dict



def convert_function_str_to_yaml(function_str: str):
    """ """
    prompt = FUNCTION_GPT_PROMPT + '\n' + function_str
    return oaiapi.chat_completion(prompt=prompt, model='gpt-4')


def count_tokens(prompt: str, model: str) -> int:
    """ Function to count the number of tokens a prompt will use """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))

