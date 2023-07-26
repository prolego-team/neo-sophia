"""
Default Python Template.
"""

from typing import Any, Dict, List, Optional, Tuple
import os

from neosophia.llmtools import dispatch as dp
from neosophia.llmtools import test_dispatch as tdp
from neosophia.llmtools import openaiapi as oaiapi

from examples import project


def main():
    """main program"""

    # setup
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    oaiapi.set_api_key(api_key)

    # prepare info

    functions = tdp.EXAMPLE_FUNCTIONS

    questions = [
        'Find up to three documents that describes the process for second mortgages.',
        'Which customers have more than $1000 in their Roth IRA?',
        'What is the current date and time?'
    ]

    for question in questions:

        print(question)

        res = dp.dispatch_prompt_llm(
            llm=lambda x: oaiapi.chat_completion(x, model='gpt-4'),
            question=question,
            functions=functions
        )

        print()

        if res is not None:
            name, params = res
            print('function name:', name)
            print('function params:', params)
            print()
            print()
        else:
            print('No response')

        res = dp.dispatch_openai_functioncall(
            model='gpt-4',
            question=question,
            functions=functions
        )

        if res is not None:
            name, params = res
            print('function name:', name)
            print('function params:', params)
            print()
            print()
        else:
            print('No response')


if __name__ == '__main__':
    main()
