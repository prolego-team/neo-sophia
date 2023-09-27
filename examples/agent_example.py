""" Example script to interact with the Agent class """
import os

import yaml
import click

import neosophia.agents.utils as autils

from examples import project

from neosophia.db import sqlite_utils as sql_utils
from neosophia.llmtools import openaiapi as oaiapi
from neosophia.agents.agent import Agent
from neosophia.agents.data_classes import Variable
from neosophia.agents.system_prompts import PARAM_AGENT_BP, TOOL_AGENT_BP

opj = os.path.join


@click.command()
@click.option('--toggle', '-t', is_flag=True, help='Toggle variables')
def main(toggle):
    """ main """
    print('\n')

    oaiapi.set_api_key(oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH))

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create a workspace for the agent to save things
    workspace_dir = autils.create_workspace_dir(config)

    # Load and/or generate tools
    tools_filepath = opj(workspace_dir, config['Agent']['tools_filename'])
    tool_descriptions = autils.setup_and_load_yaml(tools_filepath, 'tools')
    tools = autils.setup_tools(config['Tools'], tool_descriptions)
    autils.save_tools_to_yaml(tools, tools_filepath)

    # Dictionary to store all variables the Agent has access to
    variables = {}

    # Put all data from databases into Pandas DataFrame Variables
    for db_info in config['Resources']['SQLite']:
        conn = sql_utils.get_conn(db_info['path'])
        name = db_info['name']
        tables = sql_utils.get_tables_from_db(conn)

        for table in tables:

            # Don't include system tables
            if table in ['sqlite_master', 'sqlite_sequence']:
                continue

            data = sql_utils.execute_query_pd(conn, f'SELECT * FROM {table};')

            description = f'All data from the {table} {table} in database {name}\n'
            variable = Variable(
                name=f'{table}_data',
                value=data,
                description=description)

            variables[f'{table}_data'] = variable

    agent_base_prompt = TOOL_AGENT_BP
    agent = Agent(
        workspace_dir,
        TOOL_AGENT_BP,
        PARAM_AGENT_BP,
        tools,
        variables,
        toggle=toggle)
    agent.chat()


if __name__ == '__main__':
    main()
