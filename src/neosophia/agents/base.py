"""
"""
import os
import types
import sqlite3
import readline

from abc import ABCMeta, abstractmethod
from typing import Dict, List

import click

import neosophia.db.chroma as chroma
import neosophia.agents.utils as autils

from examples import project

from neosophia.db import sqlite_utils as sql_utils
from neosophia.llmtools import dispatch as dp, openaiapi as oaiapi
from neosophia.agents.system_prompts import FUNCTION_GPT_PROMPT

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
        prompt += f'Resource Info: {info}\n\n'
        self.data_resources.append(prompt)

    def add_tool(self, resource):
        self.tools.append(resource)

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

    def __init__(
            self,
            system_prompt: str,
            name: str,
            modules: List[types.ModuleType],
            resources: Dict):
        """
        - Takes as inputs a list of modules that are available to use
        - Converts each function to a function description

        """
        self.workspace = opj('.agents', f'agent_{name}')
        os.makedirs(self.workspace, exist_ok=True)

        exit()

        self.tools = tools
        self.resources = resources

        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.openai_llm_model_name = 'gpt-4'

        self.chat_history = [self.system_prompt]

    def execute(self, prompt):
        print('Thinking...')
        return oaiapi.chat_completion(
            prompt=prompt,
            model=self.openai_llm_model_name)

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

    def chat(self):

        while True:

            prompt = Prompt()

            user_input = input('> ')

            prompt.add_command(user_input)

            function_str_list = self.build_str_from_tools()
            for function_str in function_str_list:
                prompt.add_tool(function_str)

            for name, info in self.resources.items():
                prompt.add_data_resource(name, info)

            print(prompt.generate_prompt())
            exit()

    def build_str_from_tools(self):

        def _param_str(pname: str, pdesc: dp.ParamDesc) -> str:
            """
            Make a string to represent a parameter name and description.
            """
            required_str = '' if not pdesc.required else ', required'
            return f'{pname} ({pdesc.typ.__name__}{required_str}) - {pdesc.description}'

        functions = {k: v[1] for k, v in self.tools.items()}
        functions_str_list = []
        for name, desc in functions.items():
            functions_str_list.append(
                'name: ' + name + '\n' +
                'description: ' + desc.description + '\n' +
                '\n'.join([
                    'parameter: ' + _param_str(pname, pdesc)
                    for pname, pdesc in desc.params.items()
                ]) + '\n\n'
            )
        return functions_str_list


def setup():
    db_file = opj(project.DATASETS_DIR_PATH, 'synthbank.db')
    conn = sqlite3.connect(db_file)
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    return api_key, conn


def main():
    """ main """

    api_key, conn = setup()

    module_list = [sql_utils]

    function_list = [autils.build_function_str_from_module(sql_utils)[0]]

    module_dict = {}
    for fstr in function_list:
        prompt = FUNCTION_GPT_PROMPT + fstr

        description = oaiapi.chat_completion(
            prompt=prompt,
            model='gpt-4')

        print('description:', description)
        exit()

    base_agent = Agent(tools, resources)


    exit()

    db_file = opj(project.DATASETS_DIR_PATH, 'bank_database.db')
    conn = sqlite3.connect(db_file)
    sql_utils.create_database_from_csv(conn, csv_file, TABLE_NAME)
    schema = sql_utils.get_table_schema(conn, TABLE_NAME)

    print(schema, '\n-\n')

    print(sql_utils.get_tables_from_db(conn))
    exit()

    tools = {
    }

    resources = {
    }

    agent = Agent(tools, resources)
    agent.chat()


if __name__ == '__main__':
    main()

