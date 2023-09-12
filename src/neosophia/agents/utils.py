""" General utilities used by the Agent class """
import os
import re
import ast
import types
import inspect
import textwrap
import importlib

from typing import Any, Callable, Dict, List, Tuple
from dataclasses import asdict

import yaml
import tiktoken
import astunparse

import neosophia.agents.system_prompts as sp

from neosophia.db import sqlite_utils as sql_utils
from neosophia.llmtools import openaiapi as oaiapi
from neosophia.agents.data_classes import Colors, Resource, Tool

opj = os.path.join


def cprint(*args) -> None:
    """
    Prints the values of the given arguments with color formatting.

    Args:
        *args: Variable number of arguments to be printed.

    Returns:
        None
    """

    # Get the source code of the caller
    frame = inspect.currentframe().f_back
    var_names = inspect.getframeinfo(
        frame).code_context[0].strip().split('(')[1].split(')')[0].split(',')
    for idx, var in enumerate(args):
        if var == '\n':
            print(var, end='')
        else:
            print(f"{Colors.BLUE}{var_names[idx].strip()}{Colors.ENDC}: {var}")


def setup_sqlite_resources(
        sqlite_resources: List[Dict[str, str]],
        resources_filepath: str,
        workspace_resources: Dict[str, Dict[str, str]]) -> Dict[str, Resource]:

    """
    Sets up SQLite resources by adding them to a resources dictionary.

    Args:
        sqlite_resources (list): List of dictionaries representing the SQLite
        resources to be added.  Each dictionary must contain the keys 'name',
        'path', and 'description'.
        resources_filepath (str): Path to the resources yaml file.
        workspace_resources (dict): Dictionary containing existing workspace
        resources.

    Returns:
        resources (dict): Dictionary containing the added SQLite resources.
    """

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


def create_workspace_dir(config: Dict) -> str:
    """
    Create a workspace directory for the Agent.

    Args:
        config (dict): A dictionary containing the configuration settings for
        the Agent.

    Returns:
        workspace_dir (str): The path to the created workspace directory.
    """
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

    Args:
        module (types.ModuleType): The module containing the functions.
        function_names (List[str]): A list of function names to include in the
        dictionary.

    Returns:
        function_dict (Dict[str, Tuple[Callable, str]]): A dictionary mapping
        function names to tuples containing the callable function and its
        string representation.
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

    Args:
        function_str (str): The string representation of a function.

    Returns:
        str: The YAML representation of the given function string.
    """
    return oaiapi.chat_completion(
        system_prompt=sp.FUNCTION_GPT_PROMPT,
        user_prompt=function_str,
        model='gpt-4')


def parse_response(text: str) -> Dict[str, str]:
    """
    Parse the provided text into a dictionary.

    Args:
        text (str): The text to be parsed.

    Returns:
        dict: A dictionary representation of the parsed text.
    """

    keywords = ['Thoughts', 'Tool', 'Resource', 'Returned', 'Description']

    # Split the text into lines
    lines = text.split('\n')

    parsed_dict = {}
    for line in lines:

        line_start = line.split(': ', 1)[0]

        current_keyword = None
        if line_start in keywords or line_start.startswith(
                'Parameter_') or line_start.startswith(
                        'Variable_') or line_start.startswith('Resource_'):
            current_keyword = line_start
            line = ''.join(line.split(current_keyword + ': ')[1:])

        if current_keyword is not None:
            values = parsed_dict.setdefault(current_keyword, [])
            values.append(line)

    result_dict = {}
    for key, val in parsed_dict.items():

        val = '\n'.join(val)

        if key.startswith('Parameter_'):
            val = val.split(' | ')

        result_dict[key] = val

    return result_dict


def get_database_description(db_file: str, model: str = 'gpt-4') -> str:
    """
    Generate a description for a given database file using an OpenAI LLM model.

    Args:
        db_file (str): The path to the database file.
        model (str): The OpenAI LLM to use

    Returns:
        description (str): The generated description for the database.
    """

    conn = sql_utils.get_conn(db_file)
    tables = sql_utils.get_tables_from_db(conn)
    table_schemas = {}
    for table in tables:
        table_schemas[table] = sql_utils.get_table_schema(conn, table)

    user_prompt = f'Database name: {db_file}\n'
    user_prompt += 'Database Tables:\n\n'
    for table, schema in table_schemas.items():
        user_prompt += table
        user_prompt += schema.to_string() + '\n'

        query = f'SELECT * FROM {table} LIMIT 3'
        sample = sql_utils.execute_query(conn, query)
        user_prompt += f'Data Sample:\n{sample}\n\n'

    return oaiapi.chat_completion(
        system_prompt=sp.DB_INFO_PROMPT, user_prompt=user_prompt, model=model)


def count_tokens(prompt: str, model: str) -> int:
    """
    Function to count the number of tokens a prompt will use

    Args:
        prompt (str): The prompt to count the tokens of
        model (str): The model to use for token encoding

    Returns:
        count (int): The number of tokens in the prompt
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))


def setup_and_load_yaml(filepath: str, key: str) -> Dict[str, Dict[str, Any]]:
    """
    Helper function to set up workspace, create a file if it doesn't exist,
    and load data from a YAML file.

    Args:
        filepath (str): The path to the YAML file.
        key (str): The key to extract data from the loaded YAML file.

    Returns:
        dict: A dictionary containing processed data extracted from the YAML
        file. If the file doesn't exist or the data is None, an empty
        dictionary is returned.
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

    return {item['name']: item for item in data[key]}


def write_dict_to_yaml(
        data_dict: Dict, keyword: str, filename: str) -> None:
    """
    Write the given dictionary to a YAML file using a specified keyword.

    Args:
        data_dict (Dict): A dictionary containing the details to be
        written to the YAML file.
        keyword (str): The keyword to use when wrapping the dictionary in the
        YAML file.
        filename (str): The name of the YAML file to write the data to.

    Returns:
        None
    """
    # Transform the dictionary into the desired format
    data_list = [details for name, details in data_dict.items()]

    # Wrap the list in a dictionary with the keyword
    data = {keyword: data_list}

    # Write the data to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    print(f'Wrote to {filename}')


def remove_yaml_special_chars(yaml_string: str) -> str:
    """
    Remove special characters commonly used in YAML from the given string.

    Args:
        yaml_string (str): The string containing YAML to be processed.

    Returns:
        processed_string (str): The processed string with YAML special
        characters removed.
    """
    # Define a pattern for YAML special characters
    pattern = r'[:{}\[\],&*#!|>]'

    # Use regex to replace the special characters with an empty string
    return re.sub(pattern, '', yaml_string)


def process_for_yaml(name: str, description: str, width=80) -> str:
    """
    Convert the given name and description into a YAML formatted string

    Args:
        name (str): The name to be converted into a YAML string
        description (str): The description to be converted into a YAML string
        width (int): The maximum width of each line in the YAML string (default
        is 80)

    Returns:
        yaml_output (str): The YAML formatted string containing the name and
        description
    """

    # Replace double quotes with single quotes
    description = description.replace('"', '\'')

    description = remove_yaml_special_chars(description)

    # Make sure indentation is correct
    wrapper = textwrap.TextWrapper(width=width, subsequent_indent='    ')
    wrapped_description = wrapper.fill(description)
    yaml_output = f'- name: {name}\n  description: {wrapped_description}'

    return yaml_output


def save_tools_to_yaml(tools: Dict[str, Tool], filename: str) -> None:
    """
    Converts a dictionary of tools into YAML format and saves it to a file.

    Args:
        tools (dict): A dictionary containing Tools with each key being the
        tool name
        filename (str): The name of the file to save the YAML data to.

    Returns:
        None
    """

    # Convert the tools dictionary into a list of dictionaries
    tools_list = []
    for tool in tools.values():
        tool_yaml = yaml.safe_load(tool.description)
        tools_list.append(tool_yaml)

    # Wrap the list in a dictionary with the key 'tools'
    data = {'tools': tools_list}

    # Write the data to the YAML file
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)


def setup_tools(
        modules_list: List[Dict[str, Any]],
        tool_descriptions: Dict[str, Dict[str, str]]) -> Dict[str, Tool]:
    """
    This function sets up tools by generating a dictionary of Tool objects
    based on the given modules and tool descriptions.

    Args:
        modules_list (list): A list of dictionaries containing modules and the
        functions to be used from those modules.
        tool_descriptions (dict): A dictionary containing descriptions for
        specific tools.

    Returns:
        tools (dict): A dictionary containing Tools with each key being the
        tool name

    """

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
