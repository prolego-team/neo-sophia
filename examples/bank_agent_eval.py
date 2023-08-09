"""
Test the bank agent with questions and answers.
"""

from typing import Iterable, Optional, List, Tuple, Callable
import sqlite3
import time
import re
import random

import pandas as pd
import tqdm

from neosophia.llmtools import openaiapi as openai, tools
from neosophia.agents.react import make_react_agent
from neosophia.llmtools import openaiapi as oaiapi

from examples import bank_agent as ba
from examples import project

DATABASE = 'data/synthbank.db'
DEBUG = True


def main():
    """main program"""

    # configuration
    n_runs = 3

    # setup
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    oaiapi.set_api_key(api_key)

    # build stuff

    db_connection = sqlite3.connect(DATABASE)

    schema_description = ba.get_schema_description(db_connection)

    system_message = ba.get_system_message()
    system_message += schema_description
    function_descriptions = ba.FUNCTION_DESCRIPTIONS

    query_database, _ = tools.make_sqlite_query_tool(db_connection)
    functions = {
        'query_database': query_database
    }

    # Systems to evaluate. These take a question as input and return
    # an answer or None (for an uncaught error or if the system can't
    # answer the question) as well as a count of API / LLM interactions

    def build_agent(model_name: str, simple: bool) -> Callable:
        """build an agent"""

        model = openai.start_chat(model_name)

        def call(question: str) -> Tuple[Optional[str], int]:
            """answer a question with the simple agent"""
            agent = make_react_agent(
                system_message, model, function_descriptions, functions,
                ba.MAX_LLM_CALLS_PER_INTERACTION,
                simple_formatting=simple)
            return find_answer(agent(question))

        return call

    def dummy(_: str) -> Tuple[Optional[str], int]:
        """Dummy system for quickly testing things."""
        # time.sleep(random.random() * 3.0)
        time.sleep(random.random() * 0.1)
        return 'As an AI model, I\'m unable to answer the question.', 1

    systems = {
        # 'dummy': dummy,
        'agent (simple)': build_agent(model_name='gpt-4-0613', simple=True),
        'agent (react)': build_agent(model_name='gpt-4-0613', simple=False),
        'agent (simple, 3.5)': build_agent(model_name='gpt-3.5-turbo-0613', simple=True),
    }

    qs_and_evals = [
        (
            'Who most recently opened a checking account?',
            lambda x: 'John Thompson' in x
        ),
        (
            'How many people have opened a savings account in the last year?',
            lambda x: '34' in words(x)
        ),
        (
            'How many products does the person who most recently opened a mortgage have?',
            lambda x: '2' in words(x)
        ),
        (
            'Which customer has the highest interest rate on their credit card, and what is that interest rate?',
            lambda x: ('Edith Nelson' in x or '100389' in x) and ('0.3' in words(x) or '30%' in words(x))
        )
    ]

    results = {}

    for question, eval_func in tqdm.tqdm(qs_and_evals):
        for system_name, system in systems.items():
            for run_idx in range(n_runs):

                # use the system to answer the question
                start_time = time.time()
                answer, call_count = system(question)
                print('ANSWER:')
                print(answer)
                print(call_count)
                end_time = time.time()
                total_time = end_time - start_time

                info = {
                    'time': round(total_time, 3),
                    'answer': answer,
                    'calls': call_count
                }

                # evaluation
                if answer is not None:
                    info['missing'] = False
                    info['correct'] = eval_func(answer)
                else:
                    info['missing'] = True
                    info['correct'] = False

                results[(system_name, question, run_idx)] = info

    db_connection.close()

    output_file_prefix = 'eval'
    output_file_name = f'{output_file_prefix}.csv'
    header = 'system,question,run,calls,time,missing,correct,answer\n'
    with open(output_file_name, 'w') as f:
        f.write(header)
        for (system_name, question, run_idx), info in results.items():
            answer = info['answer']
            if answer is not None:
                answer = answer.replace('\n', '%%%').replace('"', '``')
            line = [
                f'"{system_name}"',
                f'"{question}"',
                run_idx,
                info['calls'],
                info['time'],
                info['missing'],
                info['correct'],
                f'"{answer}"'
            ]
            line = [str(x) for x in line]
            line = ','.join(line) + '\n'
            f.write(line)

    print(f'wrote `{output_file_name}`')

    # read csv back in with pandas to aggregate

    df = pd.read_csv(output_file_name)

    df_system_question = df.groupby(['system', 'question']).agg(
        func={'calls': 'mean', 'time': 'mean', 'missing': 'mean', 'correct': 'mean'})
    df_system = df.groupby(['system']).agg(
        func={'calls': 'mean', 'time': 'mean', 'missing': 'mean', 'correct': 'mean'})

    df_system_question.to_csv(f'{output_file_prefix}_system_question.csv', float_format='{:.3f}'.format)
    df_system.to_csv(f'{output_file_prefix}_system.csv', float_format='{:.3f}'.format)

    print('wrote aggregate CSVs')


def find_answer(messages: Iterable[openai.Message]) -> Tuple[Optional[str], int]:
    """
    Consume messages from an agent until you find 'Final Answer:'
    from the assistant.

    Note: I've seen the assistant say things like "I don't have
    enough information to answer the question, but if you give
    me the right information, I'll be able to answer and I'll make
    sure to begin my answer with 'Final Answer:'". LOLOLOL.
    """
    answer_message = None
    if DEBUG:
        print('-' * 50)

    call_count = 0

    try:
        for message in messages:
            if DEBUG:
                print('MESSAGE:')
                print('Role:', message.role)
                print('Name:', message.name)
                print('Content:', message.content)
                print('Function Call:', message.function_call)
                print('~~~')
            if message.role == 'user' or message.role == 'function':
                call_count += 1
            if message.role == 'assistant':
                # I think this logic is correct and shouldn't cause early stopping.
                if 'Final Answer:' in message.content:
                    answer_message = message
                    break
    except Exception as e:
        print('error while reading messages:', str(e))
        return None, call_count

    if answer_message is not None:
        return answer_message.content, call_count
    else:
        return None, call_count


def words(x_str: str) -> List[str]:
    """Split a string into words and strip basic punctuation from the ends."""
    res = re.split('\\s+', x_str)
    return [x.strip('.,;?-') for x in res]


if __name__ == '__main__':
    main()
