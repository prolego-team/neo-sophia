""" Agent base class """
import os
import types
import readline

from typing import Dict, List

import yaml

import neosophia.agents.utils as autils

from neosophia.llmtools import openaiapi as oaiapi

opj = os.path.join

TABLE_NAME = 'data'

TOKEN_LIMIT = {
    'gpt-4': 8192
}


class Prompt:
    """
    Represents a structured prompt that aids in constructing interactive
    conversations with the model.
    """
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
    """
    Represents an agent that interacts with the user with structured prompts to
    converse with the model.
    """
    def __init__(
            self,
            name: str,
            system_prompt: str,
            modules: List[types.ModuleType],
            resources: List[str],
            model: str = 'gpt-4'):
        """
        Initializes an Agent object
        """

        self.model_name = model
        self.token_limit = TOKEN_LIMIT[model]

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
        """ Function to give a command to interact with the LLM """

        def get_command(prompt):
            print('\nAsk a question')
            user_input = ''
            while user_input == '':
                user_input = input('> ')
            prompt.add_command(user_input)
            return user_input

        while True:

            prompt = Prompt()
            prompt.add_base_prompt(self.system_prompt)

            user_input = get_command(prompt)

            # Add resources to prompt
            for name, resource in self.resources.items():
                prompt.add_resource(name, yaml.dump(resource, sort_keys=False))

            # Add tools to prompt
            for func_description in self.tools.values():
                prompt.add_tool(yaml.dump(func_description, sort_keys=False))

            prompt_str = prompt.generate_prompt()

            # Dictionary containing the variable name:value from returned
            # function calls
            function_resources = {}
            while True:

                print(prompt_str)
                response = self.execute(prompt_str)
                print(response)

                parsed_response = autils.parse_response(response)
                function, args = self.extract_params(
                    parsed_response, function_resources)

                if function is None:
                    user_input = get_command(prompt)
                    prompt_str = prompt.generate_prompt()
                    continue

                # The LLM has enough information to answer the question
                if function == self.function_calls['extract_answer']:
                    answer = self.extract_answer(
                        user_input, function_resources)
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

                            # Add the error into the prompt so it can fix it
                            prompt_str += f'\nERROR\n{e}'
                            print('ERROR:', str(e))
                            response = self.execute(prompt_str)
                            print(response)
                            parsed_response = autils.parse_response(response)
                            function, args = self.extract_params(
                                parsed_response, function_resources)

                    if not called:
                        print('\nReached max number of function call tries\n')
                        exit()

                    # Variable name from function call
                    return_name = response.split(
                        'Returned:')[1].replace(' ', '').rstrip().strip()

                    # Add variable to function resources
                    function_resources[return_name] = res

                    prompt.add_function_resources(return_name, str(res))
                    prompt.add_completed_step(response)
                    prompt_str = prompt.generate_prompt()

                    # Get an approximate token count without needing to encode
                    num_tokens = len(prompt_str.replace(' ', '')) / 4

                    n = 1
                    # Truncate prompt if it's too long - probably a better way
                    # to do this to keep relevant information
                    while num_tokens > self.token_limit:
                        prompt_str = prompt_str[n:]
                        n += 1
                        num_tokens = autils.count_tokens(
                            prompt_str, self.model_name)

                    input('\nPress enter to continue...')

            print(answer)
            print(80 * '-')

    def extract_params(self, parsed_data: Dict, function_resources: Dict):
        """ Extract parameters from LLM response """

        func_key = 'Function'
        param_prefix = 'Parameter_'

        params = []
        values = []

        function = None
        if func_key in parsed_data:
            if parsed_data[func_key] in self.function_calls:
                function = self.function_calls[parsed_data[func_key]]

        for key, value in parsed_data.items():
            if key.startswith(param_prefix):
                param_name = value[0]
                param_value = value[1]

                if param_value in function_resources:
                    param_value = function_resources[param_value]

                param_type = value[2].replace(' ', '')

                # Stripping out quotes differently when it's a query
                if param_type == 'str' and param_name != 'query':
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
            prompt=prompt, model=self.model_name)

