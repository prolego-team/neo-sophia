""" General utilities used by the Agent class """
import os
import re
import ast
import types
import textwrap
import importlib

from typing import Any, Callable, Dict, List, Tuple
from dataclasses import dataclass

import yaml
import tiktoken
import astunparse

from neosophia.db import sqlite_utils as sql_utils
from neosophia.llmtools import openaiapi as oaiapi
from neosophia.agents.prompt import Prompt
from neosophia.agents.system_prompts import (DB_INFO_PROMPT,
                                             FUNCTION_GPT_PROMPT,
                                             NO_CONVERSATION_CONSTRAINT)

opj = os.path.join


@dataclass
class Resource:
    name: str
    path: str
    description: str

    def __str__(self):

        output = f'Resource Name: {self.name}\n'
        output += f'Path: {self.path}\n'
        output += f'Decription: {self.description}'

        return output


@dataclass
class Variable:
    name: str
    value: Any
    display_name: str
    description: str


@dataclass
class Tool:
    name: str
    function_str: str
    description: str
    call: Callable

    def __str__(self):
        output = f"Tool Name: {self.name}\n"
        output += f"Description: {self.description}\n"
        return output


def setup_sqlite_resources(
        sqlite_resources, workspace_dir, resources_filepath, workspace_resources):

    resources = {}

    # Add SQLite resources and variables
    for database in sqlite_resources:
        name = database['name']
        path = database['path']
        description = database['description']

        # If no description provided in `config.yaml`, generate one or use the
        # existing generated description from `workspace_dir/resources.yaml`
        if description is None:

            # First check if we've already generated a description in the
            # workspace directory
            resource = workspace_resources.get(name)

            if resource is None:
                # No existing description, so generate one
                print(f'Generating description for {name}...')
                description = get_database_description(path)
                resource_yaml = process_for_yaml(name, description)
                resource_data = yaml.safe_load(resource_yaml)[0]
                description = resource_data['description']

                # Save resource description to yaml file
                resources[name] = Resource(name, path, description)
                write_dict_to_yaml(
                    {
                        k: asdict(v) for k, v in resources.items()
                    },
                    'resources',
                    resources_filepath
                )
            else:
                # Load existing description
                print(f'Loading description for {name} from {resources_filepath}')
                resources[name] = Resource(name, path, resource['description'])
        else:
            # User provided a custom description which we will use
            print(f'Loading description for {name} from config.yaml')
            resources[name] = Resource(name, path, description)

    return resources


def create_workspace_dir(config):
    """ Create a workspace for the Agent """
    workspace_dir = config['Agent']['workspace_dir']
    if workspace_dir is None:
        workspace_dir = opj('.agents', config['Agent']['name'])
    os.makedirs(workspace_dir, exist_ok=True)
    return workspace_dir


def build_function_dict_from_module(
        module: types.ModuleType,
        function_names: List[str]) -> Dict[str, Tuple[Callable, str]]:
    """
    Takes a list of python modules as input and builds a dictionary containing
    the function name as the key and a tuple containing the callable function
    and its string representation as the value.
    """

    function_dict = {}
    with open(module.__file__, 'r') as f:
        function_text = ''.join(f.readlines())
    for node in ast.walk(ast.parse(function_text)):
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            callable_function = getattr(module, node.name)
            function_dict[node.name] = (
                callable_function, astunparse.unparse(node))

    return function_dict


def convert_function_str_to_yaml(function_str: str) -> str:
    """
    Convert a given function string to YAML format using a GPT-4 model.
    """
    prompt = FUNCTION_GPT_PROMPT + '\n' + function_str
    return oaiapi.chat_completion(prompt=prompt, model='gpt-4')


def parse_response(text):
    """
    Parse the provided text into a dictionary.

    Args:
        text (str): The text to be parsed.

    Returns:
        dict: A dictionary representation of the parsed text.
    """

    # Split the text into lines
    lines = text.split("\n")

    parsed_dict = {}
    for line in lines:

        # Split the line based on ": " to separate the key and the value
        parts = line.split(": ", 1)
        if len(parts) == 2:
            key, value = parts

            # Check if there are multiple '|' in the value
            if '|' in value:
                value_parts = value.split(' | ')
                parsed_dict[key] = value_parts
            else:
                parsed_dict[key] = value

    return parsed_dict


def get_database_description(db_file: str) -> str:
    """
    Generate a description for a given database file using a GPT-4 model.
    """

    conn = sql_utils.get_conn(db_file)
    tables = sql_utils.get_tables_from_db(conn)
    table_schemas = {}
    for table in tables:
        table_schemas[table] = sql_utils.get_table_schema(conn, table)

    prompt = DB_INFO_PROMPT
    prompt += f'Database name: {db_file}\n'
    prompt += 'Database Tables:\n\n'
    for table, schema in table_schemas.items():
        prompt += table
        prompt += schema.to_string() + '\n\n'

    return oaiapi.chat_completion(prompt=prompt, model='gpt-4')


def count_tokens(prompt: str, model: str) -> int:
    """ Function to count the number of tokens a prompt will use """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))


def setup_and_load_yaml(filepath, key):
    """
    Helper function to set up workspace, create a file if it doesn't exist,
    and load data from a YAML file.
    """

    # Create a file if it doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            pass

    # Load data from the file
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    # Process the loaded data
    if data is None:
        return {}
    else:
        return {item['name']: item for item in data[key]}


def write_dict_to_yaml(
        functions_dict: Dict, keyword: str, filename: str) -> None:
    """
    Write the given dictionary to a YAML file using a specified keyword.
    """

    # Transform the dictionary into the desired format
    data_list = [details for name, details in functions_dict.items()]

    # Wrap the list in a dictionary with the keyword
    data = {keyword: data_list}

    # Write the data to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    print(f'Wrote to {filename}')


def remove_yaml_special_chars(yaml_string):
    """
    Remove special characters commonly used in YAML from the given string.
    """

    # Define a pattern for YAML special characters
    pattern = r'[:{}\[\],&*#!|>]'

    # Use regex to replace the special characters with an empty string
    return re.sub(pattern, '', yaml_string)


def process_for_yaml(name: str, description: str, width=80) -> str:
    """
    Convert the given name and description into a YAML formatted string
    """

    # Replace double quotes with single quotes
    description = description.replace('"', '\'')

    description = remove_yaml_special_chars(description)

    # Make sure indentation is correct
    wrapper = textwrap.TextWrapper(width=width, subsequent_indent='    ')
    wrapped_description = wrapper.fill(description)
    yaml_output = f'- name: {name}\n  description: {wrapped_description}'
    return yaml_output


def save_tools_to_yaml(tools, filename):
    """ """

    # Convert the tools dictionary into a list of dictionaries
    tools_list = []
    for tool_name, tool in tools.items():
        tool_yaml = yaml.safe_load(tool.description)
        tools_list.append(tool_yaml)

    # Wrap the list in a dictionary with the key 'tools'
    data = {'tools': tools_list}

    # Write the data to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)


def setup_tools(modules_list, tool_descriptions):
    """ """
    tools = {}
    for module_dict in modules_list:
        module_name = module_dict['module']
        module = importlib.import_module(module_name)
        functions_list = module_dict['functions']

        function_dict = build_function_dict_from_module(
            module, functions_list)

        for func_name, (callable_func, func_str) in function_dict.items():
            if func_name in tool_descriptions:
                print(f'Loading description for {func_name}')
                description_yaml = yaml.dump(
                    tool_descriptions[func_name], sort_keys=False)
            else:
                print(f'Generating description for {func_name}')
                tool_yaml = convert_function_str_to_yaml(func_str)
                tool_dict = yaml.safe_load(tool_yaml)[0]
                description_yaml = yaml.dump(tool_dict, sort_keys=False)

            tool = Tool(
                name=func_name,
                function_str=func_str,
                description=description_yaml,
                call=callable_func
            )
            tools[func_name] = tool

    return tools
