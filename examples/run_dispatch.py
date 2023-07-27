"""
Default Python Template.
"""

from typing import Any, Dict, List, Optional, Tuple

import time

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

    dispatchers = {
        'OpenAI - Custom Prompt': lambda q, f: dp.dispatch_prompt_llm(
            llm=lambda x: oaiapi.chat_completion(x, model='gpt-4'),
            question=q,
            functions=f
        ),
        'OpenAI - Function Calling': lambda q, f: dp.dispatch_openai_functioncall(
            # model='gpt-4',
            model='gpt-4-0613',
            question=q,
            functions=f
        )
    }

    results = {}

    for question in questions:
        for disp_name, dispatcher in dispatchers.items():

            print(question)

            start_time = time.time()

            res = dispatcher(question, functions)

            end_time = time.time()

            total_time = end_time - start_time

            info = {
                'time': round(total_time, 3),
            }

            if res is not None:
                name, params = res
                print('function name:', name)
                print('function params:', params)
                print()
                print()
                info['missing'] = False
                info['name'] = name
                info['params'] = params
            else:
                print('No response')
                info['missing'] = True
                info['name'] = None
                info['params'] = None

            results[(disp_name, question)] = info

    # save results as CSV

    output_file_name = 'dispatch.csv'
    header = 'dispatcher,question,time,missing,function,params\n'

    with open(output_file_name, 'w') as f:
        f.write(header)
        for rid, info in results.items():
            line = [
                f'"{rid[0]}"',
                f'"{rid[1]}"',
                info['time'],
                info['missing'],
                info['name'],
                f"\"{info['params']}\""
            ]
            line = [str(x) for x in line]
            line = ','.join(line) + '\n'
            f.write(line)

    print(f'wrote `{output_file_name}`')


if __name__ == '__main__':
    main()
