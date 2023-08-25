""" Agent base class """
import os
import re
import sys
import time
import datetime
import readline

from typing import Dict

import neosophia.agents.utils as autils

from neosophia.llmtools import openaiapi as oaiapi
from neosophia.agents.prompt import Prompt
from neosophia.agents.data_classes import GPT_MODELS, Resource, Tool, Variable
from neosophia.agents.system_prompts import (ANSWER_QUESTION_PROMPT,
                                             CHOOSE_RESOURCES_PROMPT,
                                             CHOOSE_VARIABLES_AND_RESOURCES_PROMPT,
                                             CHOOSE_VARIABLES_PROMPT,
                                             FIX_QUERY_PROMPT,
                                             NO_CONVERSATION_CONSTRAINT,
                                             NO_TOOL_PROMPT)

opj = os.path.join


class Agent:
    """
    Represents an agent that interacts with the user with structured prompts to
    converse with the model.
    """
    def __init__(
            self,
            name: str,
            workspace_dir: str,
            system_prompt: str,
            tools: Dict[str, Tool],
            resources: Dict[str, Resource],
            variables: Dict[str, Variable],
            model_name: str = 'gpt-4-0613',
            toggle: bool = True):
        """ Initializes an Agent object """

        # Keep a log and save it to the workspace_dir
        self.log = {
            'prompt': [],
            'response': []
        }

        # Get info such as max_tokens, cost per token, etc.
        self.model_info = GPT_MODELS[model_name]

        # Monetary cost of input and output from the LLM
        self.input_cost = 0.
        self.output_cost = 0.

        self.name = name
        self.tools = tools
        self.toggle = toggle
        self.llm_calls = 0
        self.variables = variables
        self.resources = resources
        self.workspace_dir = workspace_dir
        self.system_prompt = system_prompt

        # Manually add the `extract_answer` function that's used at the end of
        # every interaction
        self.tools['extract_answer'] = Tool(
            name='extract_answer',
            function_str=None,
            description='Tool to extract an answer given a question and data',
            call=self.extract_answer
        )

        self.tools['system_exit'] = Tool(
            name='system_exit',
            function_str=None,
            description='Tool to exit the program',
            call=sys.exit
        )

    def calculate_prompt_cost(self, prompt: str):
        """ Function to calculate the input or output cost """
        num_tokens = autils.count_tokens(prompt, self.model_info.name)
        return {
            'input': num_tokens * self.model_info.input_token_cost,
            'output': num_tokens * self.model_info.output_token_cost
        }

    def _toggle_items(self, items_dict, base_prompt, command):
        """
        Helper function to toggle visibility of items (variables or resources).
        """
        prompt = Prompt()
        prompt.add_base_prompt(base_prompt)
        prompt.add_command(command)
        prompt.add_constraint(NO_CONVERSATION_CONSTRAINT)

        for item in items_dict.values():
            item.visible = False
            if isinstance(item, Variable):
                prompt.add_variable(item, True)
            elif isinstance(item, Resource):
                prompt.add_resource(item, True)

        prompt_str = prompt.generate_prompt()
        response = self.execute(prompt_str, False, False)

        items_to_show = autils.parse_response(response)
        for item_name in items_to_show.values():
            items_dict[item_name].visible = True

    def toggle_variables(self, command):
        """ Function to choose which variables to show the values for """
        self._toggle_items(self.variables, CHOOSE_VARIABLES_PROMPT, command)

    def toggle_resources(self, command):
        """ Function to choose which data resources to show """
        self._toggle_items(self.resources, CHOOSE_RESOURCES_PROMPT, command)

    def toggle_variables_and_resources(self, command: str):
        """
        Tries to toggle which variables and resources are visible to the Agent
        in a single call. If the context is too big, it splits it into two
        calls (one for the variables, one for the resources)
        """
        prompt = Prompt()
        prompt.add_base_prompt(CHOOSE_VARIABLES_AND_RESOURCES_PROMPT)
        prompt.add_command(command)
        prompt.add_constraint(NO_CONVERSATION_CONSTRAINT)

        def helper(items_dict):
            for item in items_dict.values():
                item.visible = False
                if isinstance(item, Variable):
                    prompt.add_variable(item, True)
                elif isinstance(item, Resource):
                    prompt.add_resource(item, True)

        helper(self.variables)
        helper(self.resources)

        prompt_str = prompt.generate_prompt()

        # Variables and Resources fit into one prompt
        if self.check_prompt(prompt_str):
            response = self.execute(prompt_str, False, False)
            items_to_show = autils.parse_response(response)
            for item_name in items_to_show.values():
                if item_name in self.variables:
                    self.variables[item_name].visible = True
                elif item_name in self.resources:
                    self.resources[item_name].visible = True
        else:
            self.toggle_variables(command)
            self.toggle_resources(command)

    def check_prompt(self, prompt: str):
        """ Function to check if the prompt fits in the context window """
        num_tokens = autils.count_tokens(prompt, self.model_info.name)
        if num_tokens < self.model_info.max_tokens:
            return True
        return False

    def build_prompt(self, user_input, completed_steps):
        """ Builds a prompt object and returns a string """

        prompt = Prompt()
        prompt.add_base_prompt(self.system_prompt)

        prompt.add_command(user_input)

        for step in completed_steps:
            prompt.add_completed_step(step)

        # Add resources to prompt
        for resource in self.resources.values():
            prompt.add_resource(resource)

        # Add tools to prompt
        for tool in self.tools.values():
            prompt.add_tool(tool)

        # Add variables to prompt
        for variable in self.variables.values():
            prompt.add_variable(variable)

        return prompt.generate_prompt()

    def get_running_cost(self):
        """ Returns the running input and output cost for the LLM """
        return {
            'input': self.input_cost,
            'output': self.output_cost,
            'total': self.input_cost + self.output_cost
        }

    def chat(self):
        """ Function to give a command to interact with the LLM """

        time_start = time.time()

        # Track the number of LLM calls throughout the interaction
        self.llm_calls = 0

        def get_command():
            print('\nAsk a question')
            user_input = ''
            while user_input == '':
                user_input = input('> ')
            if user_input == 'exit':
                sys.exit(1)
            return user_input

        while True:

            user_input = get_command()

            completed_steps = []

            prompt_str = self.build_prompt(user_input, completed_steps)

            # Toggle variables and resources on/off if the user wants or if the
            # prompt with all variables and resources is too long
            if self.toggle or not self.check_prompt(prompt_str):
                self.toggle_variables_and_resources(user_input)
                prompt_str = self.build_prompt(user_input, completed_steps)

            while True:

                response = self.execute(prompt_str)

                completed_steps.append(response)

                parsed_response = autils.parse_response(response)
                tool, args = self.extract_params(parsed_response)

                # Agent chose a tool that isn't available
                if tool is None:
                    completed_steps[-1] = completed_steps[-1] + NO_TOOL_PROMPT
                    prompt_str = self.build_prompt(user_input, completed_steps)
                    continue

                # The LLM has enough information to answer the question
                if tool.name == 'extract_answer':
                    answer = self.extract_answer(user_input)
                    break

                try:
                    res = tool.call(**args)
                except Exception as e:
                    # Add the error into the prompt so it can fix it
                    completed_steps[-1] = completed_steps[-1] + f'\nERROR\n{e}'
                    prompt_str = self.build_prompt(user_input, completed_steps)
                    continue

                # Variable name and description from function call
                return_name = parsed_response['Returned'].replace(
                    ' ', '').rstrip()
                description = parsed_response['Description']

                # Add variable to variables
                return_var = Variable(
                    name=return_name,
                    value=res,
                    description=description)
                self.variables[return_name] = return_var

                prompt_str = self.build_prompt(user_input, completed_steps)

                # Toggle variables and resources
                if self.toggle or not self.check_prompt(prompt_str):
                    self.toggle_variables_and_resources(user_input)
                    prompt_str = self.build_prompt(user_input, completed_steps)

                # Get an approximate token count without needing to encode
                num_tokens = int(len(prompt_str) // 4)

                n = 1
                # Truncate prompt if it's too long - probably a better way
                # to do this to keep relevant information
                while num_tokens > self.model_info.max_tokens:
                    prompt_str = prompt_str[n:]
                    n += 1
                    num_tokens = autils.count_tokens(
                        prompt_str, self.model_info.name)

            end_time = round(time.time() - time_start, 2)
            total_cost = round(self.input_cost + self.output_cost, 2)
            print(answer)
            nct = f'| Number of LLM Calls: {self.llm_calls}\n'
            nct += f'| Time: {end_time}\n'
            nct += f'| Input Cost: {round(self.input_cost, 2)}\n'
            nct += f'| Output Cost: {round(self.output_cost, 2)}\n'
            nct += f'| Total Cost: {total_cost}'

            print('\n')
            print(nct)
            print('\n')

            self.save_log()

    def extract_params(self, parsed_data: Dict):
        """ Extract parameters from LLM response """

        func_key = 'Tool'
        param_prefix = 'Parameter_'

        # Get the tool from available tools. If this function returns None,
        # then it must have chosen a tool that isn't available
        tool = None
        if func_key in parsed_data:
            if parsed_data[func_key] in self.tools:
                tool = self.tools[parsed_data[func_key]]

        # Create a dictionary of arguments to be passed to the function
        args = {}
        for key, value in parsed_data.items():
            if key.startswith(param_prefix):
                param_name = value[0]
                param_value = value[1]
                param_type = value[2].replace(' ', '')
                param_vr = value[3]

                # The parameter is a reference to a Variable in self.variables
                if 'reference' in param_vr:
                    # Strip out any quotes that might be at the beginning/end
                    if param_value[0] == "'" or param_value[0] == '"':
                        param_value = param_value[1:-1]
                    param_value = self.variables[param_value].value

                # Parameter is a string but not a SQL query
                elif param_type == 'str' and param_name != 'query':
                    param_value = str(param_value.replace("'", ""))
                    param_value = str(param_value.replace('"', ""))

                # Parameter is a SQL query that has other variables in it
                elif param_type == 'str' and param_name == 'query' and '+' in param_value:
                    param_value = self.replace_variables_in_query(param_value)

                # Add param name and its value to the dictionary
                args[param_name.replace(' ', '')] = param_value

        return tool, args

    def replace_variables_in_query(self, query):
        """
        Replace variables in the query string with their corresponding values
        from the variables dictionary.
        """

        # Regular expression pattern to identify potential variable names
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'

        # Not sure yet how this works for queries with multiple variables
        # needing replacement
        for match in re.finditer(pattern, query):
            word = match.group()
            if word in self.variables:
                prompt = Prompt()
                prompt.add_base_prompt(FIX_QUERY_PROMPT)
                for variable in self.variables.values():
                    prompt.add_variable(variable, True)

                prompt_str = prompt.generate_prompt()
                prompt_str += '\nOriginal Query: ' + query + '\n\n'
                prompt_str += 'Modified Query:'
                query = self.execute(prompt_str)

        return query

    def extract_answer(self, question):
        """ """
        prompt = Prompt()
        prompt.add_base_prompt(ANSWER_QUESTION_PROMPT)
        prompt.add_command(question)
        for variable in self.variables.values():
            if variable.visible:
                prompt.add_variable(variable)
        prompt_str = prompt.generate_prompt()
        return self.execute(prompt_str)

    def execute(self, prompt, print_prompt=True, print_response=True):
        """ """
        self.llm_calls += 1
        if print_prompt or print_response:
            print('Thinking...')
        self.input_cost += self.calculate_prompt_cost(prompt)['input']
        response = oaiapi.chat_completion(
            prompt=prompt, model=self.model_info.name)
        self.log['prompt'].append(prompt)
        self.log['response'].append(response)
        self.output_cost += self.calculate_prompt_cost(response)['output']
        if print_prompt:
            print('#' * 60 + ' PROMPT BEGIN ' + '#' * 60)
            print(prompt)
            print('#' * 60 + ' PROMPT END ' + '#' * 60)
            print('\n')
        if print_response:
            print('#' * 60 + ' RESPONSE BEGIN ' + '#' * 60)
            print(response)
            print('#' * 60 + ' RESPONSE END ' + '#' * 60)
            print('\n')
        return response

    def save_log(self):
        """
        Save the log dictionary to a text file in a readable format.

        Parameters:
        - log (dict): Dictionary containing prompts and responses.
        - filename (str): Name of the file to save the log.

        Returns:
        None
        """
        save_dir = opj(self.workspace_dir, 'logs')
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'log_{timestamp}.txt'
        filepath = opj(save_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        with open(filepath, 'w') as file:
            for prompt, response in zip(self.log['prompt'], self.log['response']):
                file.write("Prompt: " + prompt + "\n")
                file.write("Response: " + response + "\n")
                file.write("-" * 80 + "\n")
        print('Log saved to', filepath)

