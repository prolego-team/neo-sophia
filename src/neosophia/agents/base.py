"""
"""
import readline

from abc import ABCMeta, abstractmethod
from typing import Dict, List

import neosophia.db.chroma as chroma

from examples import project

from neosophia.llmtools import dispatch as dp, openaiapi as oaiapi, pdf_utils
from neosophia.db.pdfdb import PDFDB

api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)

NO_CONVERSATION_CONSTRAINT = (
    'Do not engage in conversation or provide '
    'an explanation. Simply provide an answer.')

DEFAULT_SYSTEM_PROMPT = """You are a Unified Natural Language Query chatbot
(UNLQ-GPT) and your job is to assist a user in different tasks that require
gathering and interpreting data from multiple sources. The user will provide
the task, and it is your job to come up with a plan in order to provide what is
necessary given the available resources and constraints. Each step in your plan
must be tied to an available action you can take to execute the step."""

"""Provide your output as outlined in the example below."""

"""
User input:

--------------------------COMMAND--------------------------
James Smith has applied for a mortgage loan. Determine if there are any
immediate concerns that would disqualify him from obtaining a loan.

-------------------------------DATA RESOURCES-------------------------------
- SQLite Database:

Table Name: data
Table Schema:
   cid              name type  notnull dflt_value  pk
0    0              Name             0       None   0
1    1     Date_of_Birth             0       None   0
2    2           Address             0       None   0
3    3  Checking_Account             0       None   0
4    4   Savings_Account             0       None   0
5    5          ROTH_IRA             0       None   0

- ChromaDB database

Collections: pdf_collection, page_collection, section_collection

------------------------------TOOLS-------------------------------
Function Name: sqlite_query
Function Description: 'run a SQLite query on a database'
Function Params: {'query_str': ParamDesc(description='query string', typ=<class 'str'>, required=True)}

Function Name: find_pdf_by_keyword
Function Description: Returns the Lists the IDs of a collection in a ChromaDB database
Function Params: {'name': ParamDesc(description='The collection name', typ=<class 'str'>, required=True)}
-------------------------------------------------------------------------------

UNLQ-GPT Output:

Plan:
    1.
    Action: Query the SQLite database for account information regarding James
    Smith
    Function: `sqlite_query`

    2.
    Action: Search the `pdf_collection` for any forms related to James Smith
    that may cause concern for a loan, e.g., a bankruptcy form.
    Function: `find_pdf_by_keyword`

    3.


    4.
    Action: Aggregate the collected information into a concise summary.
    Function:


    5. Determine if James Smith should be considered for a loan.
"""


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

    def __init__(self, tools: Dict, resources: Dict):

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


def build_find_pdf_by_keyword(client):

    description = dp.FunctionDesc(
        description='Lists the IDs of a collection in a ChromaDB database',
        params={
            'name': dp.ParamDesc(
                description='The collection name',
                typ=str,
                required=True
            ),
            'keyword': dp.ParamDesc(
                description='The keyword to search for',
                typ=str,
                required=True
            )
        }
    )

    def find_pdf_by_keyword(name, keyword):

    return find_pdf_by_keyword, description


tools = {
}

resources = {
}

agent = Agent(tools, resources)
agent.chat()

