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
model = openai.start_chat('gpt-3.5-turbo')
# log.info(openai.get_models_list())

log.debug('Getting the DB information.')
con = sqlite3.connect('synthbank.db')
cur = con.cursor()


cursor = con.cursor()
tables = cursor.execute('SELECT name FROM sqlite_master').fetchall()

schema_description = "The bank's database tables were created using the following commands.\n"
for table in tables:
    table_name = table[0]
    if 'autoindex' in table_name or 'sqlite_sequence' in table_name:
        continue

    description = cursor.execute(f"select sql from sqlite_master where type='table' and name='{table_name}'").fetchone()[0]
    schema_description += '  '+description+'\n'


def query_database(sqlite_query: str):
    cur = con.cursor()
    try:
        results = cur.execute(sqlite_query).fetchall()
    except:
        results = 'Query failed.'
    return results

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

system_message = \
"""You are an assistant for a retail bank.  You have the ability to run sqlite queries 
against the bank's databse to collect information for the user.\n\n"""
system_message += schema_description
log.debug(f'System message = {system_message}')
log.debug('Calling the LLM')
messages = [
    openai.Message('system', system_message),
    openai.Message('user', 'Get a list of customers who have an auto loan.')
]
log.debug('Input messages (excluding system):')
log.debug(messages[1:])
response = model(messages, functions=functions)

if response.function_call is not None:
    name = response.function_call['name']
    arguments = json.loads(response.function_call['arguments'])
    log.debug(f'LLM responded with a function call: {name}({arguments}')
    if name=='query_database':
        results = query_database(**arguments)
        log.debug('Results:')
        log.debug(results)

