""" Example script to interact with the Agent class """
import os

from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

import yaml

import neosophia.agents.utils as autils

from examples import project

from neosophia.db import sqlite_utils as sql_utils
from neosophia.agents import tools, ttt
from neosophia.agents.base import Agent
from neosophia.agents.system_prompts import UNLQ_GPT_BASE_PROMPT
from neosophia.llmtools import openaiapi as oaiapi

opj = os.path.join

RESOURCES_FILENAME = 'resources.yaml'


@dataclass
class Resource:
    name: str
    path: str
    description: str


@dataclass
class Variable:
    name: str
    value: Any
    display_name: str
    description: str


def create_workspace_dir(config):
    """ """

    # Create a workspace for the Agent
    workspace_dir = config['Agent']['workspace_dir']
    if workspace_dir is None:
        workspace_dir = opj('.agents', config['Agent']['name'])
    os.makedirs(workspace_dir, exist_ok=True)
    return workspace_dir


def setup_resources(
        config, workspace_dir, resources_filepath, workspace_resources):

    resources = {}

    # Add SQLite resources and variables
    for database in config['SQLite']:
        name = database['name']
        path = database['path']
        description = database['description']

        resources[name] = {}

        # If no description provided in `config.yaml`, generate one or use the
        # existing generated description from `workspace_dir/resources.yaml`
        if description is None:

            # First check if we've already generated a description in the
            # workspace directory
            resource = workspace_resources.get(name)

            if resource is None:
                # No existing description, so generate one
                print(f'Generating description for {name}...')
                description = autils.get_database_description(path)
                resource_yaml = autils.process_for_yaml(name, description)
                resources[name] = yaml.safe_load(resource_yaml)[0]
                description = resources[name]['description']

                # Save resource description to yaml file
                autils.write_dict_to_yaml(
                    resources, 'resources', resources_filepath)
            else:
                # Load existing description
                print(f'Loading description for {name} from {resources_filepath}')
                resources[name]['name'] = resource['name']
                resources[name]['description'] = resource['description']
        else:
            # User provided a custom description which we will use
            print(f'Loading description for {name} from config.yaml')
            resources[name]['name'] = name
            resources[name]['description'] = description

    return resources


def main():
    """ main """
    print('\n')

    oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)

    modules = [ttt]

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    workspace_dir = create_workspace_dir(config)

    variables = {}

    resources_filepath = opj(
        workspace_dir, config['Agent']['resources_filename'])

    workspace_resources = autils.setup_and_load_yaml(
        resources_filepath, 'resources')

    resources = setup_resources(
        config, workspace_dir, resources_filepath, workspace_resources)

    exit()

    conn = resource.connect(path)
    variables[name + '_conn'] = Variable(
        name + '_conn', conn, str(conn),
        f'Connection to the {name} database')
    print(variables)

    system_prompt = UNLQ_GPT_BASE_PROMPT
    agent = Agent('MyAgent', system_prompt, modules, resources, tools_dict)
    agent.chat()


if __name__ == '__main__':
    main()
