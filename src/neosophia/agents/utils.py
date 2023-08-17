""" General utilities used by the Agent class """
import os
import re
import ast
import types
import textwrap

from typing import Callable, Dict, List, Tuple

import yaml
import tiktoken

import astunparse

from neosophia.db import sqlite_utils as sql_utils
from neosophia.llmtools import openaiapi as oaiapi
from neosophia.agents.prompt import Prompt
from neosophia.agents.system_prompts import (DB_INFO_PROMPT,
                                             FUNCTION_GPT_PROMPT,
                                             NO_CONVERSATION_CONSTRAINT)

BASE_PROMPT = """You are an agent that generates SQL queries to gather data in
order to answer the given question. You have the schema of the database you are
operating on. Generate a query in the following format.

Database Name: Database to generate a query for
Query: The SQL query to run

If the database is not relevant to the question being asked, return your answer
in this format instead:

Database Name: None
Query: None
"""

def answer_question_from_databases(question: str, resources: Dict):
    """
    Function that when given a list of database connections, extracts relevant
    information in order to answer the question that was asked.
    """

    connections = {}
    db_schemas = {}
    for db_file in resources.keys():
        conn = sql_utils.get_conn(db_file)
        connections[db_file] = conn
        schema = sql_utils.get_db_creation_sql(conn)
        db_schemas[db_file] = schema

    for db_file, conn in connections.items():
        prompt = Prompt()
        prompt.add_base_prompt(BASE_PROMPT)
        prompt.add_resource(db_file, db_schemas[db_file])
        prompt.add_command(question)
        prompt.add_constraint(NO_CONVERSATION_CONSTRAINT)

        print(prompt.generate_prompt())
        print('\n==========================\n')

        response = oaiapi.chat_completion(
            prompt=prompt.generate_prompt(), model='gpt-4')
        print('response')
        print(response)


def build_function_dict_from_modules(
        modules: List[types.ModuleType]) -> Dict[str, Tuple[Callable, str]]:
    """
    Takes a list of python modules as input and builds a dictionary containing
    the function name as the key and a tuple containing the callable function
    and its string representation as the value.
    """

    function_dict = {}
    for module in modules:
        with open(module.__file__, 'r') as f:
            function_text = ''.join(f.readlines())
        for node in ast.walk(ast.parse(function_text)):
            if isinstance(node, ast.FunctionDef):
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


def setup_and_load_yaml(workspace_dir, filename, key):
    """
    Helper function to set up workspace, create a file if it doesn't exist,
    and load data from a YAML file.
    """

    # Create a workspace if it doesn't exist
    os.makedirs(workspace_dir, exist_ok=True)

    # Create a file if it doesn't exist
    file_path = os.path.join(workspace_dir, filename)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass

    # Load data from the file
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    # Process the loaded data
    if data is None:
        return {}, file_path
    else:
        return {item['name']: item for item in data[key]}, file_path


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

