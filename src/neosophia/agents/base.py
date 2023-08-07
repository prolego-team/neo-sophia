"""
"""
import os
import types
import pickle
import sqlite3
import readline

from abc import ABCMeta, abstractmethod
from typing import Dict, List

import yaml
import click

import neosophia.db.chroma as chroma
import neosophia.agents.utils as autils

from examples import project

from neosophia.db import sqlite_utils as sql_utils
from neosophia.llmtools import dispatch as dp, openaiapi as oaiapi
from neosophia.agents.system_prompts import (FUNCTION_GPT_PROMPT,
                                             UNLQ_GPT_BASE_PROMPT,
                                             UNLQ_GPT_EXAMPLE_PROMPT)

opj = os.path.join

TABLE_NAME = 'data'

NO_CONVERSATION_CONSTRAINT = (
    'Do not engage in conversation or provide '
    'an explanation. Simply provide an answer.')


class Prompt:

    def __init__(self):

        self.commands = []
        self.data_resources = []
        self.tools = []
        self.examples = []
        self.constraints = []

    def add_command(self, command):
        self.commands.append(command)

    def add_example(self, example):
        self.examples.append(example)

    def add_data_resource(self, name, info):
        prompt = f'Resource Name: {name}\n'
        prompt += f'Resource Info: {info}\n'
        self.data_resources.append(prompt)

    def add_tool(self, description):
        self.tools.append(description)

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def generate_prompt(self):
        prompt = ''
        til = 30 * '-'
        if self.commands:
            commands = '\n'.join(self.commands)
            prompt += f'{til}COMMANDS{til}\n{commands}\n\n'
        if self.tools:
            resources = '\n'.join(self.tools)
            prompt += f'{til}TOOLS{til}\n{resources}\n\n'
        if self.data_resources:
            resources = '\n'.join(self.data_resources)
            prompt += f'{til}DATA RESOURCES{til}\n{resources}\n\n'
        if self.constraints:
            constraints = '\n'.join(self.constraints)
            prompt += f'{til}CONSTRAINTS{til}\n{constraints}\n\n'
        if self.examples:
            for idx, example in enumerate(self.examples):
                prompt += f'EXAMPLE {idx + 1}:\n{example}\n'

        return prompt


class Agent:

    def __init__(self, name, system_prompt, modules, data_resources):
        """

        """

        self.openai_llm_model_name = 'gpt-4'

        self.system_prompt = system_prompt
        self.modules = modules
        self.data_resources = data_resources

        # Create a workspace for saving things
        self.workspace = opj('.agents', f'agent_{name}')
        os.makedirs(self.workspace, exist_ok=True)

        self.function_dict = autils.build_function_dict_from_modules(modules)

        # Generate a dictionary of name: Callable
        self.function_calls = {}
        for func_name, (call, func_str) in self.function_dict.items():
            self.function_calls[func_name] = call

        # Create a file to save tools (functions from modules)
        self.tools_file = opj(self.workspace, 'tools.yaml')
        if not os.path.exists(self.tools_file):
            with open(self.tools_file, 'w') as f:
                pass

        # Load any existing tools
        with open(self.tools_file, 'r') as f:
            self.tools = yaml.safe_load(f)

        if self.tools is None:
            self.tools = {}
        else:
            self.tools = {func['name']: func for func in self.tools['functions']}

        # Convert functions list to yaml format and save in tools.yaml
        for func_name, (_, func_str) in self.function_dict.items():
            if func_name not in self.tools:
                print(f'Function {func_name} not in existing tools, adding...')
                function_yaml = autils.convert_function_str_to_yaml(
                    func_str)
                self.tools[func_name] = yaml.safe_load(function_yaml)[0]

        self.write_functions_to_yaml(self.tools, self.tools_file)
        print(f'Wrote functions to {self.tools_file}')

        self.chat_history = [self.system_prompt]

    def chat(self):

        prompt = """
-------------------------COMMAND-------------------------
You are SQLInfo-GPT, an AI that generates SQL queries that can obtain schema information from an SQLite database. Use the available data resources and Python functions to generate a query that will obtain information about the database and table structure.

-------------------------TOOLS-------------------------
- name: get_tables_from_db
  description: This function retrieves a list of all table names from the database.
  params:
    conn:
      description: A connection object representing the SQLite database.
      type: sqlite3.Connection
      required: true
  returns:
    description: A list of table names.
    type: str


- name: get_db_creation_sql
  description: This function constructs a description of the database schema for the
    LLM by retrieving the CREATE commands used to create the tables.
  params:
    conn:
      description: A connection object representing the SQLite database.
      type: sqlite3.Connection
      required: true
  returns:
    description: A string that contains the schema description of the database.
    type: str

-------------------------DATA RESOURCES-------------------------
Resource Name: SQLite Database
Resource Info: data/synthbank.db"""

        print(self.execute(prompt))
        exit()

        while True:

            prompt = Prompt()

            #user_input = input('> ')
            user_input = ('Which customer has more money in their checking '
                          'account than they do in their savings account?')
            prompt.add_command(user_input)

            for resource in self.data_resources:
                prompt.add_data_resource('SQLite Database', resource)

            for func_name, func_description in self.tools.items():
                prompt.add_tool(yaml.dump(func_description, sort_keys=False))

            prompt_str = prompt.generate_prompt()

            full_prompt = self.system_prompt + '\n' + prompt_str

            print(full_prompt)
            print('TOKENS:', autils.count_tokens(full_prompt, 'gpt-4'), '\n')

            #response = self.execute(full_prompt)
            with open('response.txt', 'r') as f:
                response = f.read()

            print('\n', 50 * '-', '\n')
            print(response)

            steps = response.split('\n\n')

            exit()

    def execute(self, prompt):
        print('Thinking...')
        return oaiapi.chat_completion(
            prompt=prompt, model=self.openai_llm_model_name)

    def answer_question(self, question, context, constraint):
        prompt = Prompt()
        prompt.add_command(question)
        prompt.add_resource(context)

        if constraint is not None:
            prompt.add_constraint(constraint)

        prompt_str = prompt.generate_prompt()
        print('PROMPT')
        print(prompt_str, '\n---\n')
        print('ANSWER')
        return self.execute(prompt_str)

    def write_functions_to_yaml(self, functions_dict, filename):
        # Transform the dictionary into the desired format
        functions_list = [details for name, details in functions_dict.items()]

        # Wrap the list in a dictionary with the key 'functions'
        data = {'functions': functions_list}

        # Write the data to the YAML file
        with open(filename, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)


def setup():
    db_file = opj(project.DATASETS_DIR_PATH, 'synthbank.db')
    conn = sqlite3.connect(db_file)
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    return api_key, conn


def main():
    """
    Workflow:
        1. Convert Python modules to dispatch function strings
        2. Initialize main Agent using UNLQ_GPT_BASE_PROMPT and
        UNLQ_GPT_EXAMPLE_PROMPT
        3.
    """

    api_key, conn = setup()

    modules = [sql_utils]
    data_resources = ['data/synthbank.db', 'data/cats_and_dogs.db']

    prompt = UNLQ_GPT_BASE_PROMPT + UNLQ_GPT_EXAMPLE_PROMPT

    agent = Agent('MyAgent', prompt, modules, data_resources)
    agent.chat()

    exit()


if __name__ == '__main__':
    main()

