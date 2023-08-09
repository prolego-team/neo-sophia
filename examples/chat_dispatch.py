"""
Interactive example using dispatch.
"""

import os
import sqlite3
import functools
from typing import List, Optional, Dict

import pandas as pd

import neosophia.db.sqlite_utils as sql_utils
from neosophia.llmtools import dispatch as dp
from neosophia.llmtools import openaiapi as oaiapi

from examples import project


OPENAI_LLM_MODEL_NAME = 'gpt-4'
CSV_FILE_PATH = os.path.join(project.DATASETS_DIR_PATH, 'bank_customers.csv')
DB_FILE_PATH = os.path.join(project.DATASETS_DIR_PATH, 'bank_database.db')
TABLE_NAME = 'data'


def main():
    """main program"""

    key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    oaiapi.set_api_key(key)

    # database setup
    conn = sqlite3.connect(DB_FILE_PATH)
    sql_utils.create_database_from_csv(conn, CSV_FILE_PATH, TABLE_NAME)
    schema_str = sql_utils.get_table_schema(conn, TABLE_NAME).to_string()
    schema_str = f'TABLE NAME: {TABLE_NAME}\nTABLE SCHEMA:\n{schema_str}'

    # chat history setup
    chat_history = []

    # dictionary of function descriptions
    func_descs = {
        'sqlite_query': dp.FunctionDesc(
            description='run a SQLite query on a database with the following schema:\n' + schema_str,
            params={
                'query_str': dp.ParamDesc(
                    description='query string',
                    typ=str,
                    required=True
                )
            }
        ),
        'search_chat_history': dp.FunctionDesc(
            description='search the interaction history for questions from the user or answers from the system',
            params={
                'search_str': dp.ParamDesc(
                    description='search string',
                    typ=str,
                    required=True
                ),
                'n_results': dp.ParamDesc(
                    description='number of results to return',
                    typ=int,
                    required=True
                )
            }
        )
    }

    # dictionary of functions
    funcs = {
        'sqlite_query': functools.partial(
            sqlite_query,
            conn=conn),
        'search_chat_history': functools.partial(
            search_chat_history,
            chat_history=chat_history)
    }

    def dispatch(q_str: str) -> Optional[str]:
        """choose a function"""
        return dp.dispatch_prompt_llm(
            llm=lambda x: oaiapi.chat_completion(x, model=OPENAI_LLM_MODEL_NAME),
            question=q_str,
            functions=func_descs
        )

    while True:

        print('~~~~ QUESTION ~~~~ ~~~~ ~~~~')
        question = input('> ')

        res = dispatch(question)

        if res is None:
            print('Could not find a function to answer the question')
            continue

        name, params = res

        func = funcs.get(name)

        if func is None:
            continue

        func_res = func(**params)

        answer = answer_question(
            question,
            func_res
        )

        print('~~~~ ANSWER ~~~~ ~~~~ ~~~~')
        print(answer)

        # update chat history
        chat_history.append({'question': question, 'answer': answer})


def answer_question(question: str, context: str) -> str:
    """Answer a question given context"""

    prompt = (
        'Given the context and question, use information in the context to answer the question.\n' +
        'CONTEXT:\n\n' +
        context + '\n\n' +
        'QUESTION:\n\n' +
        question + '\n\n' +
        'Answer the question.'
    )

    return oaiapi.chat_completion(prompt, OPENAI_LLM_MODEL_NAME)


def sqlite_query(conn, query_str: str) -> str:
    """execute a sql query"""
    try:
        query_result = pd.read_sql_query(query_str, conn).to_string()
        return (
            'QUERY: ' + query_str + '\n' +
            'QUERY RESULT: ' + query_result
        )
    except Exception as e:
        print(e)
        return 'SQL query failed.'


def search_chat_history(
        chat_history: List[Dict[str, str]],
        search_str: str,
        n_results: int) -> str:
    """dead simple chat history search that finds exact matches!"""

    hist_filtered = [
        x
        for x in chat_history
        if search_str in x['question'] or search_str in x['answer']
    ]

    res = [
        (
            'question: ' + x['question'] + '\n' +
            'answer: ' + x['answer'] + '\n\n'
        )
        for x in hist_filtered
    ][:n_results]

    return '\n'.join(res)


if __name__ == '__main__':
    main()
