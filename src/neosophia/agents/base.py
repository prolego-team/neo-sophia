""" Agent base class """
import os
import re
import sys
import types
import readline

from typing import Dict, List

import yaml

import neosophia.agents.utils as autils

from neosophia.llmtools import openaiapi as oaiapi
from neosophia.agents.prompt import Prompt
from neosophia.agents.system_prompts import (ANSWER_QUESTION_PROMPT,
                                             FIX_QUERY_PROMPT)

opj = os.path.join

TOKEN_LIMIT = {
    'gpt-4': 8192
}


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

        # Track the number of LLM calls throughout the interaction
        self.llm_calls = 0

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
        self.function_calls['system_exit'] = sys.exit

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

        # Track the number of LLM calls throughout the interaction
        self.llm_calls = 0

        def get_command(prompt):
            print('\nAsk a question')
            user_input = ''
            while user_input == '':
                user_input = input('> ')
            if user_input == 'exit':
                sys.exit(1)
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
                        'Returned:')[1].replace(' ', '').rstrip()

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

                nct1 = f'| Number of LLM Calls: {self.llm_calls} |'
                nct2 = '+' + (len(nct1) - 2) * '-' + '+'
                print('\n')
                print(nct2)
                print(nct1)
                print(nct2)
                print('\n')

            print(answer)
            print(80 * '-')

    def extract_params(self, parsed_data: Dict, function_resources: Dict):
        """ Extract parameters from LLM response """

        func_key = 'Function'
        param_prefix = 'Parameter_'

        function = None
        if func_key in parsed_data:
            if parsed_data[func_key] in self.function_calls:
                function = self.function_calls[parsed_data[func_key]]

        # Create a dictionary of arguments to be passed to teh function
        args = {}
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

                if param_type == 'str' and param_name == 'query' and '+' in param_value:
                    param_value = self.replace_variables_in_query(
                        param_value, function_resources)

                # Add param name and its value to the dictionary
                args[param_name.replace(' ', '')] = param_value

        return function, args

    def replace_variables_in_query(self, query, function_resources):
        """
        Replace variables in the query string with their corresponding values
        from the function_resources dictionary.
        """

        # Regular expression pattern to identify potential variable names
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'

        # Not sure yet how this works for queries with multiple variables
        # needing replacement
        for match in re.finditer(pattern, query):
            word = match.group()
            if word in function_resources:
                prompt = Prompt()
                prompt.add_base_prompt(FIX_QUERY_PROMPT)
                for key, val in function_resources.items():
                    prompt.add_function_resources(key, val)

                prompt_str = prompt.generate_prompt()
                prompt_str += '\nOriginal Query: ' + query + '\n\n'
                prompt_str += 'Modified Query:'
                query = self.execute(prompt_str)

        return query

    def extract_answer(self, question, data):
        prompt = ANSWER_QUESTION_PROMPT
        prompt += f'Question: {question}'
        prompt += f'Data: {data}'
        print('prompt')
        print('\n===========================================\n')
        return self.execute(prompt)

    def execute(self, prompt):
        self.llm_calls += 1
        print('Thinking...')
        return oaiapi.chat_completion(
            prompt=prompt, model=self.model_name)

