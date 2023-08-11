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

from neosophia.db import sqlite_utils as sql_utils
from neosophia.llmtools import dispatch as dp, openaiapi as oaiapi

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
        self.function_resources = []

    def add_base_prompt(self, prompt):
        """ This prompt always goes at the beginning """
        self.base_prompt.append(prompt)

    def add_command(self, command):
        self.commands.append(command)

    def add_example(self, example):
        self.examples.append(example)

    def add_function_resources(self, name, value):
        prompt = f'Name: {name}\n'
        prompt += f'Value: {value}\n'
        self.function_resources.append(prompt)

    def add_resource(self, name, info):
        prompt = f'Resource Name: {name}\n'
        prompt += f'Resource Info: {info}\n'
        self.resources.append(prompt)

    def add_tool(self, description):
        self.tools.append(description)

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def add_completed_step(self, step):
        self.steps.append(step + '\n')

    def generate_prompt(self, tot=80):
        prompt = ''
        dash = '-'

        def _get_dash(text):
            n = (tot - len(text)) // 2
            extra = '' if len(text) % 2 == 0 else dash
            return dash * n + extra + text + dash * n

        def _construct(text, items):
            items = '\n'.join(items) + '\n\n'
            prompt = _get_dash(text) + '\n'
            prompt += items
            return prompt

        if self.base_prompt:
            prompt += '\n'.join(self.base_prompt)
        if self.commands:
            prompt += _construct('COMMANDS', self.commands)
        if self.tools:
            prompt += _construct('TOOLS', self.tools)
        if self.resources:
            prompt += _construct('DATA RESOURCES', self.resources)
        if self.function_resources:
            prompt += _construct('FUNCTION RESOURCES', self.function_resources)
        if self.constraints:
            prompt += _construct('CONSTRAINTS', self.constraints)
        if self.examples:
            for idx, example in enumerate(self.examples):
                prompt += f'EXAMPLE {idx + 1}:\n{example}\n'
        if self.steps:
            prompt += _construct('COMPLETED STEPS', self.steps)

        prompt += tot * dash
        return prompt


class Agent:

    def __init__(self, name, system_prompt, modules, resources, model='gpt-4'):
        """

        """

        self.openai_llm_model_name = model

        self.system_prompt = system_prompt
        self.modules = modules
        self.resources = resources

        # Create a workspace for the agent where it saves tools and resources
        self.workspace = opj('.agents', f'{name}')

        # Create or load an existing yaml file - tools and resources are dicts
        self.resources, self.resources_file = autils.setup_and_load_yaml(
            self.workspace, 'resources.yaml', 'resources')
        self.tools, self.tools_file = autils.setup_and_load_yaml(
            self.workspace, 'tools.yaml', 'functions')

        # Go through each database file and get a description from the schema
        for resource in resources:

            # Only do it for databases we haven't saved in the yaml file yet
            if resource not in self.resources:
                print(f'Adding new resource {resource}...')
                description = autils.get_database_description(resource)
                resource_yaml = autils.process_for_yaml(resource, description)
                self.resources[resource] = yaml.safe_load(resource_yaml)[0]

        # Save resource descriptions to yaml file
        autils.write_dict_to_yaml(
            self.resources, 'resources', self.resources_file)

        # Builds a dictionary containing the function name as the key and a
        # tuple containing the callable function and the entire function code
        # as a string as the value
        self.function_dict = autils.build_function_dict_from_modules(modules)

        # Generate a dictionary of {function name: Callable} from the functions
        # available in the modules passed in
        self.function_calls = {}
        for func_name, (call, func_str) in self.function_dict.items():
            self.function_calls[func_name] = call

        # Manually add the `extract_answer` function that's used at the end of
        # every interaction
        self.function_calls['extract_answer'] = self.extract_answer

        # Convert functions list to yaml format and save in tools.yaml
        for func_name, (_, func_str) in self.function_dict.items():

            # Only do it for functions we haven't saved in the yaml file yet
            if func_name not in self.tools:
                print(f'Adding new function {func_name}...')
                function_yaml = autils.convert_function_str_to_yaml(
                    func_str)
                self.tools[func_name] = yaml.safe_load(function_yaml)[0]

        # Save function descriptions to yaml file
        autils.write_dict_to_yaml(self.tools, 'functions', self.tools_file)

    def chat(self):

        while True:

            prompt = Prompt()

            prompt.add_base_prompt(self.system_prompt)

            print('\nAsk a question')
            #user_input = input('> ')
            user_input = 'What is the name of the customer with the oldest checking account?'
            prompt.add_command(user_input)

            for name, resource in self.resources.items():
                prompt.add_resource(name, yaml.dump(resource, sort_keys=False))

            for func_name, func_description in self.tools.items():
                prompt.add_tool(yaml.dump(func_description, sort_keys=False))

            prompt_str = prompt.generate_prompt()

            self.function_resources = {}
            while True:
                print(prompt_str)
                response = self.execute(prompt_str)
                print(response)

                parsed_response = autils.parse_response(response)
                function, args = self.extract_params(parsed_response)

                if function == self.function_calls['extract_answer']:
                    answer = self.extract_answer(
                        user_input, self.function_resources)
                    break

                else:
                    called = False
                    num_tries = 0
                    while not called and num_tries < 10:
                        try:
                            res = function(**args)
                            called = True
                        except Exception as e:
                            num_tries += 1
                            prompt_str += f'\nERROR\n{e}'
                            print('ERROR:', str(e))
                            response = self.execute(prompt_str)
                            print(response)
                            parsed_response = autils.parse_response(response)
                            function, args = self.extract_params(
                                parsed_response)

                    if not called:
                        print('\nReached max number of function call tries\n')
                        exit()

                    return_name = response.split(
                        'Returned:')[1].replace(' ', '').rstrip().strip()

                    self.function_resources[return_name] = res

                    prompt.add_function_resources(return_name, str(res))
                    prompt.add_completed_step(response)

                    prompt_str = prompt.generate_prompt()

                    input('\nPress enter to continue...')

            print('Done')
            print(answer)
            exit()

    def extract_params(self, parsed_data):
        func_key = 'Function'
        param_prefix = 'Parameter_'

        params = []
        values = []

        function = None
        if func_key in parsed_data:
            function = self.function_calls[parsed_data[func_key]]

        for key, value in parsed_data.items():
            if key.startswith(param_prefix):
                param_name = value[0]
                param_value = value[1]

                if param_value in self.function_resources:
                    param_value = self.function_resources[param_value]

                param_type = value[2].replace(' ', '')
                if param_type == 'str':
                    param_value = str(param_value.replace("'", ""))
                    param_value = str(param_value.replace('"', ""))

                params.append(param_name.replace(' ', ''))
                values.append(param_value)

        args = dict(zip(params, values))
        return function, args

    def extract_answer(self, question, data):
        prompt = 'Answer the question given the following data\n'
        prompt += f'Question: {question}'
        prompt += f'Data: {data}'
        return self.execute(prompt)

    def execute(self, prompt):
        print('Thinking...')
        return oaiapi.chat_completion(
            prompt=prompt, model=self.openai_llm_model_name)

