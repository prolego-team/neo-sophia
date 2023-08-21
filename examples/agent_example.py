""" Example script to interact with the Agent class """
import os

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import yaml

import neosophia.agents.utils as autils

from examples import project

from neosophia.db import sqlite_utils as sql_utils
from neosophia.llmtools import openaiapi as oaiapi
from neosophia.agents.base import Agent
from neosophia.agents.system_prompts import UNLQ_GPT_BASE_PROMPT

opj = os.path.join

RESOURCES_FILENAME = 'resources.yaml'


def main():
    """ main """
    print('\n')

    oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create a workspace for the agent to save things
    workspace_dir = autils.create_workspace_dir(config)

    # Load and/or generate tools
    tools_filepath = opj(workspace_dir, config['Agent']['tools_filename'])
    tool_descriptions = autils.setup_and_load_yaml(tools_filepath, 'tools')
    tools = autils.setup_tools(config['Tools'], tool_descriptions)
    autils.save_tools_to_yaml(tools, tools_filepath)

    # Load and/or generate resources
    resources_filepath = opj(
        workspace_dir, config['Agent']['resources_filename'])

    # Resources loaded from the yaml file saved in the Agent's workspace
    workspace_resources = autils.setup_and_load_yaml(
        resources_filepath, 'resources')

    resources = autils.setup_sqlite_resources(
        config['Resources']['SQLite'], workspace_dir,
        resources_filepath, workspace_resources)

    variables = {}

    # Connect to any SQLite databases
    for db_info in config['Resources']['SQLite']:
        conn = sql_utils.get_conn(db_info['path'])
        name = db_info['name']
        var_name = name + '_conn'

        variable = autils.Variable(
            name=var_name,
            value=conn,
            display_name=conn,
            description=f'Connection to {name} database')

        variables[var_name] = variable

    system_prompt = UNLQ_GPT_BASE_PROMPT
    agent = Agent('MyAgent', system_prompt, tools, resources, variables)
    agent.chat()


if __name__ == '__main__':
    main()
