""" Example script to interact with the Agent class """
from examples import project

from neosophia.db import sqlite_utils as sql_utils
from neosophia.agents import tools, ttt
from neosophia.agents.base import Agent
from neosophia.agents.system_prompts import UNLQ_GPT_BASE_PROMPT
from neosophia.llmtools import openaiapi as oaiapi


def main():
    """ main """

    oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)

    modules = [ttt]
    resources = [
        'data/synthbank.db',
        'data/cats_and_dogs.db',
    ]

    synth_conn = sql_utils.get_conn(resources[0])
    catdog_conn = sql_utils.get_conn(resources[1])

    query_synth_database = tools.build_query_database(synth_conn)
    query_catdog_database = tools.build_query_database(catdog_conn)

    cats_desc = {
        'name': 'query_synth_database',
        'description': 'Function to query the synthbank database',
        'params':
            {
                'query': {
                    'description': 'The SQL query to run',
                    'type': 'str',
                    'required': True
                }
            },
        'returns': {
            'description': 'The output from the SQL query that was run',
            'type': 'pd.DataFrame'
        }
    }

    dogs_desc = {
        'name': 'query_dogs_database',
        'description': 'Function to query the cats and dogs database',
        'params':
            {
                'query': {
                    'description': 'The SQL query to run',
                    'type': 'str',
                    'required': True
                }
            },
        'returns': {
            'description': 'The output from the SQL query that was run',
            'type': 'pd.DataFrame'
        }
    }

    tools_dict = {
        'query_synth_database': (query_synth_database, cats_desc),
        'query_dogs_database': (query_catdog_database, dogs_desc)
    }

    system_prompt = UNLQ_GPT_BASE_PROMPT
    agent = Agent('MyAgent', system_prompt, modules, resources, tools_dict)
    agent.chat()


if __name__ == '__main__':
    main()
