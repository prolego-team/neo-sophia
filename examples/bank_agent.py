import os
import sys
from collections.abc import Callable
import logging
import sqlite3
import json

from neosophia.llmtools import (
    openaiapi as openai,
    tools
)

from neosophia.agents.react import make_react_agent, make_simple_react_agent
from neosophia.db.sqlite_utils import get_db_creation_sql

# === Basic setup ===================================================
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logging.getLogger('agent').setLevel(logging.DEBUG)
log = logging.getLogger('agent')
# ===================================================================

DATABASE = 'data/synthbank.db'

# # Defina a function to query the database.  The LLM will be "told" about this
# # function.  (TODO don't return string on failure, so something better.)
# def query_database(sqlite_query: str):
#     cur = con.cursor()
#     try:
#         results = cur.execute(sqlite_query).fetchall()
#     except Exception as e:
#         results = f'Query failed: {e}'
#     return results

# Following is how GPT wants to be told what functions are available and their
# arguments.



def main():
    # Setup
    openai.set_api_key(os.getenv('OPENAI_API_KEY'))

    # Get a model
    model = openai.start_chat('gpt-4-0613')

    # Connect to the DB and get the table names
    log.debug('Getting the DB information.')
    db_connection = sqlite3.connect(DATABASE)

    cursor = db_connection.cursor()
    tables = cursor.execute('SELECT name FROM sqlite_master').fetchall()

    # Build the functions that the agent can use
    query_database, _ = tools.make_sqlite_query_tool(db_connection)
    functions = {
        'query_database': query_database
    }
    function_descriptions = [
        {
            'name': 'query_database',
            'description': 'Query the bank sqlite database.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'A sqlite query to run against the bank databse.'
                    },
                },
                'required': ['sqlite_query']
            }
        }
    ]


    # Construct a description of the DB schema for the LLM by retrieving the
    # CREATE commands used to create the tables.
    schema_description = (
        "Each customer has one or more products at the bank.  Each product has a globally unique "
        "account number.  Each customer has a globally unique guid identifier.  The customer guids "
        "and the product account numbers are related in the 'products' database table.\n\n"
        "The bank's database tables were created using the following commands.\n"
    )
    schema_description += get_db_creation_sql(db_connection)
    # for table in tables:
    #     table_name = table[0]
    #     if 'autoindex' in table_name or 'sqlite_sequence' in table_name:
    #         continue

    #     description = cursor.execute(f"select sql from sqlite_master where type='table' and name='{table_name}'").fetchone()[0]
    #     schema_description += '  '+description+'\n'

    system_message = (
        "You are an assistant for a retail bank.  You have the ability to run sqlite queries "
        "against the bank's databse to collect information for the user.  Answer the user's "
        "questions as best as you can.  Only use the functions you have been provided with."
    )
    system_message += schema_description
    agent = make_simple_react_agent(system_message, model, function_descriptions, functions)

    input_msg = input('>')
    messages = agent(input_msg)

    print('SUMMARY')
    for i,message in enumerate(messages[1:]):
        print(f'MESSAGE {i}')
        print(f'{message.role}:')
        print(message.content)
        print(f'<name={message.name}, function_call={message.function_call}>')
        print('='*20)
    # print('='*40)

    # transcription = ''
    # for message in messages:
    #     transcription += f'{message.role}: {message.content}\n'
    #     transcription += f'OpenAI API call metadata: `{{"name": "{message.name}", "function_call": "{message.function_call}}}"`\n'

    # system_message = (
    #     "You are an assistant that provides feedback to another AI assistant.  You should provide feedback to improve future "
    #     "interactions between a user and the other AI assistant."
    # )
    # user_message = (
    #     "Below is a transcription of an interaction I had with an AI assistant.  How could this interaction have been improved?\n\n"
    # )
    # user_message += transcription

    # messages = [openai.Message('system', system_message), openai.Message('user', user_message)]
    # response = model(messages, functions=functions)
    # print('='*40)
    # # print(messages)
    # print(response.content)




if __name__=='__main__':
    main()
