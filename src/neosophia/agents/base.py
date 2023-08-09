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

        self.base_prompt = []
        self.commands = []
        self.resources = []
        self.tools = []
        self.examples = []
        self.constraints = []
        self.steps = []
        self.function_stuff = []

    def add_base_prompt(self, prompt):
        """ This prompt always goes at the beginning """
        self.base_prompt.append(prompt)

    def add_command(self, command):
        self.commands.append(command)

    def add_example(self, example):
        self.examples.append(example)

    def add_function_stuff(self, name, value):
        prompt = f'Name: {name}\n'
        prompt += f'Value: {value}\n'
        self.function_stuff.append(prompt)

    def add_resource(self, name, info):
        prompt = f'Resource Name: {name}\n'
        prompt += f'Resource Info: {info}\n'
        self.resources.append(prompt)

    def add_tool(self, description):
        self.tools.append(description)

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def add_completed_step(self, step):
        self.steps.append(step)

    def generate_prompt(self):
        prompt = ''
        til = 30 * '-'
        if self.base_prompt:
            prompt += '\n'.join(self.base_prompt)
        if self.commands:
            commands = '\n'.join(self.commands)
            prompt += f'{til}COMMANDS{til}\n{commands}\n\n'
        if self.tools:
            resources = '\n'.join(self.tools)
            prompt += f'{til}TOOLS{til}\n{resources}\n\n'
        if self.resources:
            resources = '\n'.join(self.resources)
            prompt += f'{til}DATA RESOURCES{til}\n{resources}\n\n'
        if self.function_stuff:
            resources = '\n'.join(self.function_stuff)
            prompt += f'{til}FUNCTION RESOURCES{til}\n{resources}\n\n'
        if self.constraints:
            constraints = '\n'.join(self.constraints)
            prompt += f'{til}CONSTRAINTS{til}\n{constraints}\n\n'
        if self.examples:
            for idx, example in enumerate(self.examples):
                prompt += f'EXAMPLE {idx + 1}:\n{example}\n'
        if self.steps:
            steps = '\n\n'.join(self.steps)
            prompt += f'{til}COMPLETED STEPS{til}\n{steps}\n\n'
        prompt += 30 * '-'
        return prompt


class Agent:

    def __init__(self, name, system_prompt, modules, resources):
        """

        """

        self.openai_llm_model_name = 'gpt-4'

        self.system_prompt = system_prompt
        self.modules = modules
        self.resources = resources

        self.workspace = opj('.agents', f'{name}')
        self.resources, self.resources_file = autils.setup_and_load_yaml(
            self.workspace, 'resources.yaml', 'resources')
        self.tools, self.tools_file = autils.setup_and_load_yaml(
            self.workspace, 'tools.yaml', 'functions')

        # Go through each database file and get a description from the schema
        for resource in resources:
            if resource not in self.resources:
                description = autils.get_database_description(resource)
                resource_yaml = autils.process_for_yaml(resource, description)
                print(resource_yaml)
                self.resources[resource] = yaml.safe_load(resource_yaml)[0]

        # Save resource descriptions to yaml file
        autils.write_dict_to_yaml(
            self.resources, 'resources', self.resources_file)

        self.function_dict = autils.build_function_dict_from_modules(modules)

        # Generate a dictionary of name: Callable
        self.function_calls = {}
        for func_name, (call, func_str) in self.function_dict.items():
            self.function_calls[func_name] = call
        self.function_calls['extract_answer'] = self.extract_answer

        # Convert functions list to yaml format and save in tools.yaml
        for func_name, (_, func_str) in self.function_dict.items():
            if func_name not in self.tools:
                print(f'Function {func_name} not in existing tools, adding...')
                function_yaml = autils.convert_function_str_to_yaml(
                    func_str)
                self.tools[func_name] = yaml.safe_load(function_yaml)[0]

        autils.write_dict_to_yaml(self.tools, 'functions', self.tools_file)

        self.chat_history = [self.system_prompt]

    def chat(self):

        while True:

            prompt = Prompt()

            prompt.add_base_prompt(self.system_prompt)

            #user_input = input('> ')
            user_input = ('Who most recently opened a checking account?')
            user_input = 'What is the name of the oldest customer?'
            prompt.add_command(user_input)

            for name, resource in self.resources.items():
                prompt.add_resource(name, yaml.dump(resource, sort_keys=False))

            for func_name, func_description in self.tools.items():
                prompt.add_tool(yaml.dump(func_description, sort_keys=False))

            prompt_str = prompt.generate_prompt()

            #print('TOKENS:', autils.count_tokens(prompt_str, 'gpt-4'), '\n')

            self.function_returns = {}
            while True:
                print(prompt_str)
                response = self.execute(prompt_str)
                print(response)

                function, args = self.extract_params(response)

                print('function:', function)
                if function == self.function_calls['extract_answer']:
                    answer = self.extract_answer(
                        user_input, self.function_returns)

                else:
                    called = False
                    while not called:
                        print('ARGS:', args)
                        try:
                            res = function(**args)
                            #print('RES:', res)
                            called = True
                        except Exception as e:
                            prompt_str += '\nERROR\n'
                            prompt_str += str(e)
                            print('ERROR')
                            print(e)
                            response = self.execute(prompt_str)# + '\nERROR\n' + str(e))
                            function, args = self.extract_params(response)

                    return_name = response.split('Returned:')[1].replace(' ', '')
                    #print('return_name:', return_name)

                    self.function_returns[return_name] = res

                    prompt.add_function_stuff(return_name, str(res))
                    prompt.add_completed_step(response)

                    prompt_str = prompt.generate_prompt()

                    #input('\nPress enter to continue...')

            print('Done')
            print('Answer:')
            print(answer)

            exit()

    def extract_answer(self, question, data):
        prompt = 'Answer the question given the following data\n'
        prompt += f'Question: {question}'
        prompt += f'Data: {data}'
        return self.execute(prompt)

    def extract_params(self, response):
        func_prefix = 'Function: '
        param_prefix = 'Parameter_'

        params = []
        values = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith(func_prefix):
                function = self.function_calls[
                    line.removeprefix(func_prefix)]
            elif line.startswith(param_prefix):
                line = ''.join(line.split(':')[1:])
                param_name = line.split('|')[0]
                param_value = line.split('|')[1][1:-1]

                if param_value in self.function_returns:
                    param_value = self.function_returns[param_value]
                else:
                    param_value = line.split('|')[1][1:-1]

                param_type = line.split('|')[2].replace(' ', '')
                if param_type == 'str':
                    param_value = str(param_value.replace("'", ""))
                params.append(param_name.replace(' ', ''))
                values.append(param_value)

        args = dict(zip(params, values))
        return function, args

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


def setup():
    db_file = opj(project.DATASETS_DIR_PATH, 'synthbank.db')
    conn = sqlite3.connect(db_file)
    #print(sql_utils.get_tables_from_db(conn))
    #print(sql_utils.get_table_schema(conn, 'savings_account'))
    #print(sql_utils.execute_query(conn, 'SELECT * FROM customers LIMIT 2'))
    #print(sql_utils.execute_query(conn, 'SELECT * FROM savings_account LIMIT 2'))
    #exit()
    conn = create_cats_and_dogs_db('data/cats_and_dogs.csv')
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    return api_key, conn


def create_cats_and_dogs_db(csv_file: str):
    db_file = opj(project.DATASETS_DIR_PATH, 'cats_and_dogs.db')
    conn = sqlite3.connect(db_file)
    sql_utils.create_database_from_csv(conn, csv_file, 'pets')
    return conn


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
    resources = ['data/synthbank.db', 'data/cats_and_dogs.db']

    prompt = UNLQ_GPT_BASE_PROMPT# + UNLQ_GPT_EXAMPLE_PROMPT

    agent = Agent('MyAgent', prompt, modules, resources)
    agent.chat()

    exit()

    print(prompt)
    print('\n-----------\n')
    out = agent.execute(prompt)
    print('OUT')
    print(out)
    exit()
    agent.chat()

    exit()


if __name__ == '__main__':
    main()

