"""
Classes and functions for custom-built function-calling
functionality.
"""

from typing import List, Dict, Any, Tuple

from dataclasses import dataclass


@dataclass
class ParamDesc:
    """Parameter description."""
    description: str
    typ: Any
    required: bool


@dataclass
class FunctionDesc:
    """Function description"""
    description: str
    params: Dict[str, ParamDesc]


def make_prompt(
        question: str,
        functions: Dict[str, FunctionDesc]
    ) -> str:
    """aldsjkfaldfjhadf"""

    def _param_str(pname: str, pdesc: ParamDesc) -> str:
        """aldsfjahldsjf"""
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
        'FUNCTION NAME: function_name]\n' +
        '[parameter name 0]: [parameter value 0]\n' +
        '[parameter name 1]: [parameter value 1]\n' +
        '...\n'
    )

