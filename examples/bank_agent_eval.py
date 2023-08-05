"""
Test the bank agent with questions and answers.
"""
import string
from typing import Iterable, Optional, List
import sqlite3
import tqdm
import time
import re


from neosophia.llmtools import openaiapi as openai, tools
from neosophia.agents.react import make_react_agent

from examples import bank_agent as ba

from examples import project
from neosophia.llmtools import openaiapi as oaiapi

DATABASE = 'data/synthbank.db'


def main():
    """main program"""

    # configuration
    n_runs = 3

    # setup
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    oaiapi.set_api_key(api_key)

    # build stuff
    model = openai.start_chat('gpt-4-0613')

    db_connection = sqlite3.connect(DATABASE)

    system_message = ba.get_system_message()
    function_descriptions = ba.FUNCTION_DESCRIPTIONS

    query_database, _ = tools.make_sqlite_query_tool(db_connection)
    functions = {
        'query_database': query_database
    }

    # TODO: additional system functions to evaluate

    def agent_simple(question: str) -> Optional[str]:
        """answer a question with the simple agent"""
        agent = make_react_agent(
            system_message, model, function_descriptions, functions,
            ba.MAX_LLM_CALLS_PER_INTERACTION, True)
        return find_answer(agent(question))

    def agent_react(question: str) -> Optional[str]:
        """answer a question with the react agent"""
        agent = make_react_agent(
            system_message, model, function_descriptions, functions,
            ba.MAX_LLM_CALLS_PER_INTERACTION, False)
        return find_answer(agent(question))

    systems = {
        'agent (simple)': agent_simple,
        # 'agent (react)': agent_react)
    }

    qs_and_evals = [
        ('Who most recently opened a checking account?', lambda x: 'John Thompson' in x),
        ('How many people have opened a savings account in the last year?', lambda x: '34' in words(x)),
        ('How many products does the person who most recently opened a mortgage have?', lambda x: '2' in words(x)),
        (
            'Which customer has the highest interest rate on their credit card, and what is that interest rate?',
            lambda x: ('Edith Nelson' in x or '77' in x) and (('0.3' in words(x) or '30%' in words(x)))
        )
    ]

    results = {}

    for question, eval_func in tqdm.tqdm(qs_and_evals):
        for system_name, system in systems.items():
            for run_idx in range(n_runs):

                # use the system to answer the question
                start_time = time.time()
                answer = system(question)
                print('ANSWER:')
                print(answer)
                end_time = time.time()
                total_time = end_time - start_time

                info = {
                    'time': round(total_time, 3),
                    'answer': answer
                }

                # evaluation
                if answer is not None:
                    info['missing'] = False
                    info['correct'] = eval_func(answer)
                else:
                    info['missing'] = True
                    info['correct'] = False

                results[(system_name, question, run_idx)] = info

    # save results as CSV

    output_file_name = 'eval.csv'
    header = 'system,question,run,time,missing,correct,answer\n'

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
                info['time'],
                info['missing'],
                info['correct'],
                f'"{answer}"'
            ]
            line = [str(x) for x in line]
            line = ','.join(line) + '\n'
            f.write(line)

    print(f'wrote `{output_file_name}`')

    db_connection.close()

    # TODO: do some aggregation of results across runs


def find_answer(messages: Iterable[openai.Message]) -> Optional[str]:
    """
    Consume messages from an agent until you find 'Final Answer:'
    from the assistant.

    Note: I've seen the assistant say things like "I don't have
    enough information to answer the question, but if you give
    me the right information, I'll be able to answer and I'll make
    sure to begin my answer with 'Final Answer:'". LOLOLOL.
    """
    answer_message = None
    for message in messages:
        # print("MESSAGE")
        # print(message)
        # print('----')
        if message.role == 'assistant':
            if 'Final Answer:' in message.content:
                answer_message = message
                break
        # print('continuing')
    if answer_message is not None:
        return answer_message.content
    else:
        return None


def words(x_str: str) -> List[str]:
    """split a string into words"""
    res = re.split('\\s+', x_str)
    return [x.strip(string.punctuation) for x in res]


if __name__ == '__main__':
    main()
