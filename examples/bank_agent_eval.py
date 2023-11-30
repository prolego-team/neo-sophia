"""
Test the bank agent with questions and answers.
"""

from typing import Iterable, Optional, List, Tuple, Callable, Dict
import sqlite3
import time
import re
import random
import os
import datetime

import llama_cpp
import pandas as pd
import tqdm

from neosophia.llmtools import openaiapi as openai, tools
from neosophia.agents import react, react_chat
from neosophia.agents import react
from neosophia.llmtools import openaiapi as oaiapi
from neosophia.llmtools import dispatch as dp
from neosophia.db import sqlite_utils

from neosophia.agents import simplelocal
from neosophia.llmtools import promptformat as pf
from examples import project

from examples import bank_agent as ba


GPU_ENABLE = False

DATABASE = 'data/synthbank.db'
DEBUG = True


# An alternate format message that asks the system to not engage in conversation.
FORMAT_MESSAGE_QUIET = (
    "When the user asks a question, think about what to do before responding. "
    "Briefly share your thoughts, but do not engage in conversation. "
    "You can use a function call to get additional information from the user. "
    "When you have the final answer say, \"Final Answer: \" followed by the "
    "response to the user's question."
)

FUNCTION_DESCS = {
    'query_database': dp.FunctionDesc(
        description='Query the bank sqlite database.',
        params={
            'query': dp.ParamDesc(
                description='A sqlite query to run against the bank databse.',
                typ=str,
                required=True
            )
        }
    )
}

FUNCTION_DESCS_GET_SCHEMA = {
    'get_table_schema': dp.FunctionDesc(
        description='Get the schema of a sqlite table.',
        params={
            'name': dp.ParamDesc(
                description='Name of table.',
                typ=str,
                required=True
            )
        }
    )
}

QS_AND_EVALS = [
    (
        'What is the name of the customer who most recently opened a checking account?',
        lambda x: 'John Thompson' in x
    ),
    (
        'How many unique people opened a savings account between 2022-08-01 and 2023-08-01?',
        lambda x: '32' in words(x)
    ),
    (
        'How many products does the person who most recently opened a mortgage have?',
        lambda x: '2' in words(x) or 'two' in words(x) or 'Two' in words(x)
    ),
    (
        'How many customers were born between 1981 and 1996?',
        lambda x: '357' in words(x)
    ),

    # there are multiple customers with the same highest interest rate
    # (
    #     'Which customer has the highest interest rate on their credit card, and what is that interest rate?',
    #     lambda x: ('Edith Nelson' in x or '100389' in x) and ('0.3' in words(x) or '30%' in words(x))
    # ),

    (
        (
            'What are the names of the customers with the highest interest rate on their credit card, ' +
            'and what is that interest rate?'
        ),
        lambda x: (
            all(
                [name in x for name in [
                    'Edith Nelson',
                    'Kevin Mcgann',
                    'Celeste Walker',
                    'Harry Schille',
                    'Martha Alvarez',
                    'Mark Aguilar',
                    'Mark May',
                    'Maria Holmes',
                    'Mary Bell',
                    'Jeff Gill',
                    'Gerald Aldridge',
                    'Brooke Peterson',
                    'Penny Hernandez',
                    'Dennis Norwood',
                    'Pam Riegel'
                ]]
            ) and
            (('0.3' in words(x)) or '30%' in words(x) or '(0.3)' in words(x) or '(30%)' in words(x))
        )
    ),
    (
        'What is the name of the customer who has the largest mortgage loan?',
        lambda x: 'Roberta Freeman' in x
    ),
    (
        # This may not work as well asked "how many unique proudcts"
        # Might be a nice one to include
        'How many unique product types are available to customers?',
        lambda x: '7' in words(x)
    ),
    (
        'How many different checking account types are there and what are they?',
        lambda x: (
            ('2' in words(x)) and
            all(
                [name in x for name in [
                    'PremierAccess Checking Account',
                    'EasyAccess Checking Account'
                ]]
            )
        )
    )
]


def main():
    """main program"""

    # configuration
    n_runs = 3

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir_path = f'eval_{timestamp}'
    os.makedirs(output_dir_path, exist_ok=False)

    # setup
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    oaiapi.set_api_key(api_key)

    # build stuff

    db_connection = sqlite3.connect(DATABASE)

    schema_description = ba.get_schema_description(db_connection)

    system_message = ba.get_system_message(
        datetime.datetime.strptime('2023-08-31', '%Y-%m-%d'))
    system_message += schema_description

    query_database, _ = tools.make_sqlite_query_tool(db_connection)
    functions = {
        'query_database': query_database
    }

    def get_table_schema(name: str) -> str:
        return sqlite_utils.get_table_schema(db_connection, name).to_string()

    def build_query_db_debug(conn):
        def query_db_debug(query: str) -> str:
            print(f'\t\tQUERY STRING: ```{query}```')
            try:
                res = pd.read_sql_query(query, conn).to_string()
            except Exception as error:
                res = f'Query failed: {error}'
            return res
        return query_db_debug

    functions_with_get_schema = {
        # 'query_database': query_database,
        'query_database': build_query_db_debug(db_connection),
        'get_table_schema': get_table_schema
    }

    # Systems to evaluate. These take a question and seed as input and return
    # an answer or None (for an uncaught error or if the system can't
    # answer the question) as well as a count of API / LLM interactions

    def build_agent_simple_llama2(model_file_name: str, promptformat: Callable, stop: List[str]) -> Callable:
        """build an agent using a local llama model"""

        def call(question: str, seed: int) -> Tuple[Optional[str], int]:
            """answer a question with the react agent"""

            model_path = os.path.join(project.MODELS_DIR_PATH, model_file_name)
            assert os.path.isfile(model_path), 'mmodel file not found'

            llama_model = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=10000 if GPU_ENABLE else 0,
                n_ctx=simplelocal.LLAMA2_MAX_TOKENS,
                seed=seed  # AFAIK this is the only way to set the seed.
            )

            llama_model_wrapped = simplelocal.build_llama2_wrapper(
                llama_model, promptformat, stop)

            question_uid = hash(question) % 1000
            debug_func = simplelocal.build_debug_func(
                prefix=f'{model_file_name}_{question_uid}_seed_{seed}',
                promptformat=promptformat
            )

            agent = simplelocal.make_simple_agent(
                system_message=system_message,
                dp_message=simplelocal.CUSTOM_DISPATCH_PROMPT_PREFIX,
                model=llama_model_wrapped,
                function_descriptions={**FUNCTION_DESCS, **FUNCTION_DESCS_GET_SCHEMA},
                functions=functions_with_get_schema,
                max_llm_calls=ba.MAX_LLM_CALLS_PER_INTERACTION,
                debug_func=debug_func
            )
            return find_answer(agent(question))
        return call

    def build_agent(model_name: str, simple: bool) -> Callable:
        """build an agent that uses the OpenAI API"""
        model = openai.start_chat(model_name)
        def call(question: str, seed: int) -> Tuple[Optional[str], int]:
            """
            Answer a question with the simple agent.
            The seed is ignored.
            """
            if simple:
                agent = react_chat.make_react_agent(
                    system_message, model, ba.FUNCTION_DESCRIPTIONS, functions,
                    ba.MAX_LLM_CALLS_PER_INTERACTION)
            else:
                agent = react.make_react_agent(
                    system_message, model, ba.FUNCTION_DESCRIPTIONS, functions,
                    ba.MAX_LLM_CALLS_PER_INTERACTION)
            return find_answer(agent(question))
        return call

    def dummy(question: str, seed: int) -> Tuple[Optional[str], int]:
        """Dummy system for quickly testing things."""
        time.sleep(random.random() * 0.1)
        return 'As an AI model, I\'m unable to answer the question.', 1

    systems = {
        'dummy': dummy,
        'simple (gpt-4)': build_agent(model_name='gpt-4-0613', simple=True),

        # 'simple (gpt-4, quiet)': patch_agent(
        #     build_agent(model_name='gpt-4-0613', simple=True),
        #     lambda: patch_format_message_simple(format_message_quiet),
        #     undo_patch_format_message_simple
        # ),
        # 'react (gpt-4)': build_agent(model_name='gpt-4-0613', simple=False),
        # 'simple (gpt-3.5)': build_agent(model_name='gpt-3.5-turbo-0613', simple=True),

        # 'simple (llama2-13b)': build_agent_simple_llama2(
        #     'llama-2-13b-chat.q4_0.gguf',
        #     pf.messages_to_llama2_prompt,
        #     pf.STOP_LLAMA
        # ),
        #

        # ~~~~

        # 'simple (wizardcoder-34B)': build_agent_simple_llama2(
        #     'wizardcoder-python-34b-v1.0.Q4_K_M.gguf',
        #     pf.messages_to_alpaca_prompt,
        #     pf.STOP_ALPACA
        # ),

        # 'simple (wizardcoder-13B)': build_agent_simple_llama2(
        #     'wizardcoder-python-13b-v1.0.Q4_K_M.gguf',
        #     pf.messages_to_alpaca_prompt,
        #     pf.STOP_ALPACA
        # )

        # 'simple (codellama-oasst-13B)': build_agent_simple_llama2(
        #     'codellama-13b-oasst-sft-v10.Q4_K_M.gguf',
        #     pf.messages_to_chatml_prompt,
        #     pf.STOP_CHATML
        # )

        # 'simple (codellama-34B)': build_agent_simple_llama2(
        #     'codellama-34b-instruct.Q4_K_M.gguf',
        #     pf.messages_to_llama2_prompt,
        #     pf.STOP_LLAMA
        # )

        'simple (phind-codellama-34B)': build_agent_simple_llama2(
            'phind-codellama-34b-v2.Q4_K_M.gguf',
            pf.messages_to_phind_prompt,
            pf.STOP_PHIND
        )
    }

    results = eval_systems(systems, QS_AND_EVALS, n_runs)

    db_connection.close()

    output_file_prefix = 'eval'
    output_file_path = os.path.join(output_dir_path, f'{output_file_prefix}.csv')
    header = 'system,question,run,calls,time,missing,correct,answer\n'
    with open(output_file_path, 'w') as f:
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

    print(f'wrote `{output_file_path}`')

    # read csv back in with pandas to aggregate

    df = pd.read_csv(output_file_path)

    df_system_question = df.groupby(['system', 'question']).agg(
        func={'calls': 'mean', 'time': 'mean', 'missing': 'mean', 'correct': 'mean'})
    df_system = df.groupby(['system']).agg(
        func={'calls': 'mean', 'time': 'mean', 'missing': 'mean', 'correct': 'mean'})

    df_system_question.to_csv(
        os.path.join(
            output_dir_path,
            f'{output_file_prefix}_system_question.csv'
        ),
        float_format='{:.3f}'.format
    )
    df_system.to_csv(
        os.path.join(
            output_dir_path,
            f'{output_file_prefix}_system.csv',
        ),
        float_format='{:.3f}'.format
    )

    print('wrote aggregate CSVs')


def eval_systems(
        systems: Dict[str, Callable],
        qs_and_evals: List[Tuple[str, Callable]],
        n_runs: int
        ) -> Dict[Tuple[str, str, int], Dict]:
    """
    Evaluate the systems (functions that return an answer and # of LLM calls)
    using a list of pairs of question and evaluation function.
    """
    results = {}

    for question, eval_func in tqdm.tqdm(qs_and_evals):
        for system_name, system in systems.items():
            for run_idx in range(n_runs):
                # use the system to answer the question
                start_time = time.time()
                answer, call_count = system(question, seed=run_idx)
                print('ANSWER:')
                print(answer)
                print('~~~~ ~~~~ ~~~~ ~~~~')
                print('call count:', call_count)
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

    return results


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
                print('~~~ ~~~ ~~~ ~~~ ~~~')
            if message.role == 'system' or message.role == 'user' or message.role == 'function':
                call_count += 1
            if message.role == 'assistant':
                # I think this logic is correct and shouldn't cause early stopping.
                if 'final answer' in message.content.lower() and message.function_call is None:
                    answer_message = message
                    break
    except Exception as e:
        print('error while reading messages:', str(e))
        return None, call_count

    if answer_message is not None:
        return answer_message.content, call_count
    else:
        return None, call_count


def patch_agent(
        agent: Callable,
        apply_patch: Callable,
        undo_patch: Callable) -> Callable:
    """Make a new agent by patching, running, and then undoing the changes"""
    def call(question: str) -> Tuple[Optional[str], int]:
        apply_patch()
        res = agent(question)
        undo_patch()
        return res
    return call


def patch_format_message_simple(msg: str):
    """replace the default format message"""
    react_chat.FORMAT_MESSAGE_BACKUP = react_chat.FORMAT_MESSAGE
    react_chat.FORMAT_MESSAGE = msg


def undo_patch_format_message_simple():
    """undo the format message replacement"""
    react_chat.FORMAT_MESSAGE = react_chat.FORMAT_MESSAGE_BACKUP


def words(x_str: str) -> List[str]:
    """Split a string into words and strip basic punctuation from the ends."""
    res = re.split('\\s+', x_str)
    return [x.strip('.,;?-') for x in res]


if __name__ == '__main__':
    main()
