import os
import sys
import logging
import sqlite3
import json

import neosophia.llmtools.openaiapi as openai

# === Basic setup ===================================================
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logging.getLogger('agent').setLevel(logging.DEBUG)
log = logging.getLogger('agent')
# ===================================================================

openai.set_api_key(os.getenv('OPENAI_API_KEY'))
# model = openai.start_chat('gpt-3.5-turbo')
model = openai.start_chat('gpt-3.5-turbo')

# Connect to the DB and get the table names
log.debug('Getting the DB information.')
con = sqlite3.connect('synthbank.db')
cursor = con.cursor()
tables = cursor.execute('SELECT name FROM sqlite_master').fetchall()

# Construct a description of the DB schema for the LLM by retrieving the
# CREATE commands used to create the tables.
schema_description = "The bank's database tables were created using the following commands.\n"
for table in tables:
    table_name = table[0]
    if 'autoindex' in table_name or 'sqlite_sequence' in table_name:
        continue

    description = cursor.execute(f"select sql from sqlite_master where type='table' and name='{table_name}'").fetchone()[0]
    schema_description += '  '+description+'\n'

# Defina a function to query the database.  The LLM will be "told" about this
# function.  (TODO don't return string on failure, so something better.)
def query_database(sqlite_query: str):
    cur = con.cursor()
    try:
        results = cur.execute(sqlite_query).fetchall()
    except:
        results = 'Query failed.'
        raise
    return results

# Following is how GPT wants to be told what functions are available and their
# arguments.
functions = [
    {
        'name': 'query_database',
        'description': 'Query the bank sqlite database.',
        'parameters': {
            'type': 'object',
            'properties': {
                'sqlite_query': {
                    'type': 'string',
                    'description': 'A sqlite query to run against the bank databse.'
                },
            },
            'required': ['sqlite_query']
        }
    }
]

# Prepare to call the LLM
system_message = \
"""You are an assistant for a retail bank.  You have the ability to run sqlite queries 
against the bank's databse to collect information for the user.

Each customer has one or more products at the bank.  Each product has a globally unique
account number.  Each customer has a globally unique guid identifier.  The customer guids
and the product account numbers are related in the "products" database table.\n\n"""
system_message += schema_description
system_message += \
"""

In your interactions use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of the function calls available to you
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I have now answered the question
Final Answer: the final answer to the original input question"
"""
messages = [
    openai.Message('system', system_message),
    # openai.Message('user', 'How many customers have opened a new checking account in the last two years?')
    # openai.Message('user', 'What functions do you have at your disposal?')
]

log.debug(f'System message = {system_message}')

while (input_msg:=input(">"))!='':
    input_msg = f'Question: {input_msg}\nThought: '
    message = openai.Message('user', input_msg)
    log.debug('Calling the LLM')
    response = model(messages, functions=functions)
    log.debug(f'Response content = \n{response.content}')
    if response.function_call is not None:
        name = response.function_call['name']
        arguments = json.loads(response.function_call['arguments'])
        log.debug(f'LLM responded with a function call: {name}({arguments}')
        if name=='query_database':
            results = query_database(**arguments)
            results = f'Observation: {str(results)}'
            message_output = openai.Message.from_function_call(name, results)
            messages.append(message_output)
            log.debug('Function results:')
            log.debug(messages[-1])

    while "Final Answer" not in response.content:
        log.debug('Another call to the LLM')
        response = model(messages, functions=functions)
        log.debug(response)
        input('continue?')
        print()

