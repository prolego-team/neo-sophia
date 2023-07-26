"""
Classes and functions for custom-built function-calling
functionality.
"""
import json
from typing import  Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass

from neosophia.llmtools import openaiapi as oaiapi


@dataclass
class ParamDesc:
    """Parameter description."""
    description: str
    typ: Any
    required: bool


@dataclass
class FunctionDesc:
    """Function description."""
    description: str
    params: Dict[str, ParamDesc]


def dispatch_prompt(
        question: str,
        functions: Dict[str, FunctionDesc]
        ) -> str:
    """
    Build a prompt for an LLM to choose between functions
    to answer a question.
    """

    def _param_str(pname: str, pdesc: ParamDesc) -> str:
        """
        Make a string to represent a parameter
        name and description.
        """
        required_str = '' if not pdesc.required else ', required'
        return f'{pname} ({pdesc.typ.__name__}{required_str}) - {pdesc.description}'

    functions_str = ''
    for name, desc in functions.items():
        functions_str += (
            'name: ' + name + '\n' +
            'description: ' + desc.description + '\n' +
            'parameters:\n' +
            '\n'.join([
                '- ' + _param_str(pname, pdesc)
                for pname, pdesc in desc.params.items()]
            )
        )

    return (
        'Answer the question by choosing a function and generating parameters for the function ' +
        'based on the function descriptions below.\n\n'
        'QUESTION: ' + question + '\n\n' +
        'FUNCTION DESCRIPTIONS:' + '\n\n' +
        functions_str + '\n\n' +
        'Your answer should be in this form:\n\n' +
        'FUNCTION: [function_name]\n' +
        'PARAMETER: [parameter name 0] [parameter value 0]\n' +
        'PARAMETER: [parameter name 1] [parameter value 1]\n' +
        '...\n'
    )


def parse_dispatch_response(
        response: str,
        functions: Dict[str, Any]
        ) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse an LLMs response to the above dispatch prompt."""
    lines = response.split('\n')

    func_prefix = 'FUNCTION: '
    param_prefix = 'PARAMETER: '

    name = None
    params = {}

    for line in lines:
        if line.startswith(func_prefix):
            name = line.removeprefix(func_prefix)
        elif line.startswith(param_prefix):
            line = line.removeprefix(param_prefix).strip()
            # word at start of line is paramter name
            words = line.split()
            pname = words[0]
            value = line.removeprefix(pname).strip()
            params[pname] = value

    # Need to think more about the best day to handle potential issues here

    res = {}

    desc = functions.get(name)
    if desc is None:
        print(f'Error parsing function name: `{name}`')
        return None

    for pname, value in params.items():
        try:
            typ = desc.params[pname].typ
            value = typ(value)
            if isinstance(value, str):
                # There is probably a better way to do this
                value = value.strip('\'')
                value = value.strip('"')
        except Exception as e:
            print(f'Error parsing parameter `{pname}={value}')
            print(e)
        res[pname] = value

    return name, res


def dispatch_prompt_llm(
        llm: Callable,
        question: str,
        functions: Dict[str, FunctionDesc]
        ) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Use an LLM to choose a function."""

    prompt = dispatch_prompt(
        question=question,
        functions=functions
    )
    response = llm(prompt)

    return parse_dispatch_response(
        response=response,
        functions=functions
    )


def dispatch_openai_functioncall(
        model: str,
        question: str,
        functions: Dict[str, FunctionDesc],
        ) -> Optional[Tuple[str, Dict[str, Any]]]:
    """aldskfjhaldfjkhaf"""

    # TODO: type map

    type_to_name = {
        str: 'string',
        int: 'integer'
    }

    fdicts = []
    for name, desc in functions.items():
        fdict = {
            'type': 'object',
            'name': name,
            'description': desc.description,
            'parameters': {
                'type': 'object',
                'properties': {
                    pname: {
                        'type': type_to_name.get(pdesc.typ),
                        'description': str(pdesc.description)
                    }
                    for pname, pdesc in desc.params.items()
                },
                'required': [
                    pname
                    for pname, pdesc in desc.params.items()
                    if pdesc.required
                ]
            }
        }
        fdicts.append(fdict)

    chat = oaiapi.start_chat(model)

    prompt = (
        'Answer the following question with a function call.\n\n' +
        'QUESTION: ' + question
    )

    response = chat(
        messages=[
            oaiapi.Message(
                role='user',
                content=prompt
            )
        ],
        functions=fdicts
    )

    if response.function_call is None:
        print('no function call in response!')
        return None

    # TODO: check for additional issues here?

    function_call = response.function_call
    name = function_call['name']
    params = json.loads(function_call['arguments'])

    return name, params
