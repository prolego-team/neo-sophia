"""
Run sort of an integration test to compare various dispatch functions.
"""

from typing import Any, Dict, List, Optional, Tuple

import time
import os
from typing import Callable

from neosophia.llmtools import dispatch as dp
from neosophia.llmtools import test_dispatch as tdp
from neosophia.llmtools import openaiapi as oaiapi

from examples import project


def main():
    """main program"""

    llama_models_dir_path = '/Users/ben/Prolego/code/llama.cpp'

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

    # add local llama-2 dispatchers

    for name, path in [
            ('llama-2-13b-chat - Custom Prompt', 'llama-2-13b-chat.ggmlv3.q4_0.bin'),
            # ('llama-2-13b - Custom Prompt', 'llama-2-13b.ggmlv3.q4_0.bin')
            ]:
        model_file_path = os.path.join(llama_models_dir_path, path)
        dispatcher = build_llama2_dispatcher(model_file_path)
        if dispatcher is not None:
            dispatchers[name] = dispatcher

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
            param_str = str(info['params'])
            param_str = param_str.replace("\"", '``')
            line = [
                f'"{rid[0]}"',
                f'"{rid[1]}"',
                info['time'],
                info['missing'],
                info['name'],
                f"\"{param_str}\""
            ]
            line = [str(x) for x in line]
            line = ','.join(line) + '\n'
            f.write(line)

    print(f'wrote `{output_file_name}`')


def build_llama2_dispatcher(model_file_path: str) -> Optional[Callable]:
    """build a llama2 dispatcher"""

    try:
        from neosophia.llmtools import llama
        tokens = 1024
        llama_model = llama.load_llama2(
            model_file_path=model_file_path,
            context_tokens=tokens
        )
        llama_dispatcher = lambda q, f: dp.dispatch_prompt_llm(
            llm=lambda x: llama.llama2_text(llama_model, x, tokens),
            question=q,
            functions=f
        )
        return llama_dispatcher
    except Exception:
        print("Could not load llama-2 chat model")
        return None


if __name__ == '__main__':
    main()
