"""
Test the bank agent with questions and answers.
"""

import os
import sys
import logging
import sqlite3
from datetime import datetime

from typing import Optional


import gradio as gr

from neosophia.llmtools import openaiapi as openai, tools
from neosophia.agents.react import make_react_agent
from neosophia.db.sqlite_utils import get_db_creation_sql

from examples import bank_agent as ba

from examples import project
from neosophia.llmtools import openaiapi as oaiapi

DATABASE = 'data/synthbank.db'


def main():
    """main program"""

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

    agent = make_react_agent(
        system_message, model, function_descriptions, functions)

    # question = 'Who has the most money?'
    question = 'Who has the most money in their checking account?'

    answer_message = None
    for message in agent(question):
        print(message)
        if message.role == 'assistant':
            if 'Final Answer:' in message.content:
                answer_message = message
                break

    print('FINAL ANSWER:')
    if answer_message is not None:
        print(answer_message.content)
    else:
        print('UNABLE TO ANSWER THE QUESTION')

    db_connection.close()


# def find_answer(chat_history) -> Optional[str]:
#     final_message = chat_history[-1][1]
#     if final_message is None:
#         response = None
#     elif 'Final Answer:' in final_message:
#         response = final_message.split('Final Answer:')[1].strip()
#         response = response.split('\n')[0].strip()
#     else:
#         response = None
#
#     return response


if __name__ == '__main__':
    main()
