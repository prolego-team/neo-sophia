""" Agent base class """
import os
import re
import sys
import time
import datetime
import readline
import traceback

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import neosophia.agents.utils as autils
import neosophia.agents.system_prompts as sp

from neosophia.llmtools import openaiapi as oaiapi
from neosophia.agents.prompt import Prompt
from neosophia.agents.data_classes import GPT_MODELS, Tool, Variable

opj = os.path.join

# Description for extract_answer
EXTRACT_ANSWER_DESCRIPTION = (
    'Tool Name: extract_answer\n'
    'Description: This function extracts an answer given a question and '
    'a dataframe containing the answer.\n'
    'params:\n'
    '  data:\n'
    '    description: A DataFrame or Series that contains the answer\n'
    '    required: true'
)


class Log:
    """ Simple class for logging """

    def __init__(self, workspace_dir):
        """ Create the log list and save directory """
        self.log = []
        self.save_dir = opj(workspace_dir, 'logs')
        os.makedirs(self.save_dir, exist_ok=True)

    def add(self, agent_name: str, message: str) -> None:
        """
        Adds a message to the log.

        Args:
            agent_name (str): The source of the message
            message (str): The message to log
        Returns:
            None
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log.append(f'{timestamp} - [{agent_name}] {message}')

    def save(self) -> None:
        """
        Save the log to a text file in a readable format.

        Args:
            workspace_dir (str): The Agent's workspace directory

        Returns:
            None
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'log_{timestamp}.txt'
        filepath = opj(self.save_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as file:
            for log in self.log:
                file.write(log)
                file.write('\n')
        print('Log saved to', filepath)

    def __str__(self):
        return '\n'.join(self.log)


class Agent:
    """
    Represents an agent that interacts with the user with structured prompts to
    converse with the model.
    """
    def __init__(
            self,
            workspace_dir: str,
            tool_bp: str,
            param_bp: str,
            tools: Dict[str, Tool],
            variables: Dict[str, Variable],
            model_name: str = 'gpt-4-0613',
            toggle: bool = True) -> None:
        """
        Initializes an Agent object.

        Args:
            workspace_dir (str): The directory where the agent's log will be
            saved.
            tool_bp (str): The base prompt for the Tool agent.
            param_bp (str): The base prompt for the Param agent.
            tools (Dict[str, Tool]): A dictionary of tools available to the
            agent.
            variables (Dict[str, Variable]): A dictionary of variables
            available to the agent.
            model_name (str, optional): The name of the GPT model to use.
            toggle (bool, optional): A toggle for the agent. If set to True,
            the Agent will make additionall LLM calls to toggle which Variables
            are needed for the current step in execution. If False, then all
            Variables will be included in the Prompt.

        Returns:
            None
        """

        # Keep a log and save it to the workspace_dir
        self.log = Log(workspace_dir)
        self.save_dir = opj(workspace_dir, 'logs')

        self.all_steps = []

        if model_name not in GPT_MODELS:
            models = ', '.join(list(GPT_MODELS.keys()))
            print(
                f'\nModel name "{model_name}" must be one of: {models}\n')
            sys.exit(1)

        # Get info such as max_tokens, cost per token, etc.
        self.model_name = model_name
        self.model_info = GPT_MODELS[self.model_name]

        # Monetary cost of input and output from the LLM
        self.input_cost = 0.
        self.output_cost = 0.

        self.tools = dict(tools)
        self.toggle = toggle
        self.llm_calls = 0
        self.variables = dict(variables)
        self.workspace_dir = workspace_dir
        self.tool_bp = tool_bp
        self.param_bp = param_bp

        # Manually add the `extract_answer` function that's used at the end of
        # every interaction
        self.tools['extract_answer'] = Tool(
            name='extract_answer',
            function_str=None,
            description=EXTRACT_ANSWER_DESCRIPTION,
            call=self.extract_answer
        )

        self.tools['system_exit'] = Tool(
            name='system_exit',
            function_str=None,
            description='Tool to exit the program',
            call=sys.exit
        )

    def toggle_variables(self, command: str, thoughts: str) -> None:
        """
        Function to choose which variables to show the values for

        Args:
            command (str): the command from the user that will determine which
            variables to toggle

        Returns:
            None
        """
        prompt = Prompt()
        prompt.add_base_prompt(sp.CHOOSE_VARIABLES_PROMPT)
        prompt.add_command(command)
        prompt.add_command(thoughts)
        prompt.add_constraint(sp.NO_CONVERSATION_CONSTRAINT)

        for variable in self.variables.values():
            if not variable.dynamic:
                variable.visible = False
                if isinstance(variable, Variable):
                    prompt.add_variable(variable, True)
            else:
                self.log.add(
                    'TOGGLE AGENT',
                    f'Setting dynamic variable {variable.name} VISIBLE')

        prompt_str = prompt.generate_prompt()
        response = self.execute(prompt_str, False, False)

        variables_to_show = autils.parse_response(response)
        for variable_name in variables_to_show.values():
            self.log.add(
                'TOGGLE AGENT', f'Setting variable {variable_name} VISIBLE')
            self.variables[variable_name].visible = True

    def check_prompt(self, prompt: str) -> bool:
        """
        Function to check if the prompt fits in the context window

        Args:
            prompt (str): The prompt to be checked.

        Returns:
            True if the prompt fits within the context window, False otherwise.
        """
        num_tokens = autils.count_tokens(prompt, self.model_info.name)
        print(num_tokens)
        if num_tokens < self.model_info.max_tokens:
            return True
        return False

    def get_running_cost(self) -> Dict[str, float]:
        """
        Returns the running input, output, and total cost for the LLM

        Args:
            None
        Returns:
            cost_dict (dict): Dictionary contaning the input, output, and total
            cost
        """

        return {
            'input': self.input_cost,
            'output': self.output_cost,
            'total': self.input_cost + self.output_cost
        }

    def build_tool_prompt(
            self,
            base_prompt: str,
            command: str,
            completed_steps: List[str]) -> str:
        """
        Builds a prompt for choosing a Tool

        Args:
            command (str): The question/command given by the user
            completed_steps (list): The completed_steps taken by the Agent

        Returns:
            prompt (str): The generated prompt string.
        """

        prompt = Prompt()
        prompt.add_base_prompt(base_prompt)

        prompt.add_command(command)

        for step in completed_steps:
            prompt.add_completed_step(step)

        # Add tools to prompt
        for tool in self.tools.values():
            prompt.add_tool(tool)

        # Add variables to prompt
        for variable in self.variables.values():
            prompt.add_variable(variable)

        return prompt.generate_prompt()

    def build_param_prompt(
            self,
            command: str,
            parsed_response: Dict[str, Any],
            completed_steps: List[str]) -> str:
        """
        Builds a prompt for choosing parameters

        Args:
            command (str): The question/command given by the user
            parsed_response (Dict): The parsed response from the Tool agent
            completed_steps (list): The completed_steps taken by the Agent

        Returns:
            prompt (str): The generated prompt string.
        """

        tool_name = parsed_response['Tool']
        thoughts = parsed_response['Thoughts']

        prompt = Prompt()
        prompt.add_base_prompt(self.param_bp)

        # Add the user's question along with the main Agent's thoughts as
        # commands
        prompt.add_command(command)
        prompt.add_command(thoughts)

        # Add tool to prompt
        prompt.add_tool(self.tools[tool_name])

        # Add variables to prompt
        for variable in self.variables.values():
            prompt.add_variable(variable)

        for step in completed_steps:
            prompt.add_completed_step(step)

        prompt.add_constraint(sp.NO_CONVERSATION_CONSTRAINT)

        for constraint in sp.PARAM_PROMPT_CONSTRAINTS:
            prompt.add_constraint(constraint)

        return prompt.generate_prompt()

    def log_response(
            self,
            parsed_response: Dict[str, Union[str, list]],
            agent_name: str) -> None:
        """
        Adds the parsed response from the Agent to the log

        Args:
            parsed_response (Dict): The response from the agent parsed into a
            dictionary
            agent_name (str): The name of the Agent

        Returns:
            None
        """
        for key, val in parsed_response.items():
            if isinstance(val, list):
                val = ' | '.join(val)
            self.log.add(f'{agent_name}', f'{key}: {val}')

    def choose_tool(self, command: str, completed_steps: List[str]):
        """
        Chooses an appropriate tool based on the given command and prior steps.
        This function repeatedly prompts for tool selection until a valid tool
        from the available tools (`self.tools`) is chosen. If an invalid tool
        is selected, it logs an error message and prompts the Agent again.

        Args:
            command (str): The command based on which a tool is to be chosen.
            completed_steps (List[str]): A list of steps that have been
            completed prior to the invocation of this function.

        Returns:
            tuple: A tuple containing:
                - tool (Tool): The chosen tool object from `self.tools`.
                - tool_response (str): The raw response from the tool
                  selection prompt.
                - parsed_tool_response (dict): The parsed response from the
                  tool selection prompt.
        """

        tool = None
        tool_failure_steps = []
        while tool is None:

            # Choose a Tool
            tool_prompt = self.build_tool_prompt(
                self.tool_bp,
                command,
                completed_steps + tool_failure_steps)

            tool_response = self.execute(tool_prompt)
            parsed_tool_response = autils.parse_response(tool_response)
            self.log_response(parsed_tool_response, 'TOOL AGENT')

            if parsed_tool_response['Tool'] in self.tools:
                tool = self.tools[parsed_tool_response['Tool']]
            else:
                tool_name = parsed_tool_response['Tool']
                error_msg = f'{tool_name} not in TOOLS. '
                error_msg += 'Choose a tool from the TOOLS section'
                step = {
                    'status': 'error',
                    'message': error_msg
                }
                tool_failure_steps.append(step)
                self.all_steps.append(step)
                self.log.add(
                    'TOOL AGENT', f'ERROR - Wrong Tool chosen: {tool_name}')
                self.log.add('TOOL AGENT', error_msg)

        return tool, tool_response, parsed_tool_response

    def interact(self, command) -> None:

        self.log.add('SYSTEM', f'User Command: {command}')

        completed_steps = []
        self.thoughts = ''
        self.answer = None
        while True:

            if self.toggle:
                self.toggle_variables(command, self.thoughts)

            res = self.choose_tool(command, completed_steps)

            tool, tool_response, parsed_tool_response = res
            tool_name = parsed_tool_response['Tool']

            self.thoughts = str(parsed_tool_response['Thoughts'])
            if tool_name == 'execute_pandas_query':
                self.thoughts += f' {sp.NO_PYTHON}'
            completed_step = (
                'Thoughts: ' + self.thoughts + '\n' +
                'Tool: ' + str(parsed_tool_response['Tool']) + '\n'
            )

            step = {
                'status': 'success',
                'message': completed_step
            }
            completed_steps.append(step)
            self.all_steps.append(step)

            self.gradio_output = completed_step

            yield

            if parsed_tool_response['Tool'] in [
                    'system_exit', 'extract_answer']:

                # Choose parameters for the Tool
                param_prompt = self.build_param_prompt(
                    command,
                    parsed_tool_response,
                    completed_steps)
                param_response = self.execute(param_prompt).split('\n\n')
                param_response = param_response[0]
                parsed_param_response = autils.parse_response(
                    param_response)
                self.log_response(parsed_param_response, 'PARAM AGENT')

                if 'Parameter_0' not in parsed_param_response:

                    error_msg = 'You must generate a Parameter'
                    step = {
                        'status': 'error',
                        'message': error_msg
                    }
                    completed_steps.append(step)
                    self.all_steps.append(step)
                    self.log.add('SYSTEM', error_msg)
                    continue

                # This is the variable that contains the answer
                variable_name = autils.strip_quotes(
                    parsed_param_response['Parameter_0'][1])

                # Could happen if the Parameter Agent used a Python
                # expression as the data to pass to extract_answer
                if variable_name not in self.variables:

                    error_msg = (
                        'The Variable you have chosen '
                        f'`{variable_name}` is not an available Variable'
                    )
                    print(error_msg)
                    step = {
                        'status': 'error',
                        'message': error_msg
                    }
                    completed_steps.append(step)
                    self.all_steps.append(step)

                    self.log.add('SYSTEM', error_msg)
                    continue

                self.answer = self.extract_answer(
                    command, self.thoughts, data=self.variables[variable_name])

                parsed_answer = self.answer
                if 'Question 1:' in self.answer:
                    parsed_answer = self.answer.split('Question 1:')[0].rstrip()
                answer_step = f'Previous question: {command}\n'
                answer_step += f'Previous answer: {parsed_answer}'

                completed_steps = [
                    {
                        'status': 'success',
                         'message': answer_step
                    }
                ]

                answer_step += str(self.answer) + '\n'
                self.log.add('SYSTEM', answer_step)

                yield

                break

            called = False
            while not called:

                # Choose parameters for the Tool
                param_prompt = self.build_param_prompt(
                    command,
                    parsed_tool_response,
                    completed_steps)
                param_response = self.execute(param_prompt).split('\n\n')
                param_response = param_response[0]

                parsed_param_response = autils.parse_response(
                    param_response)
                self.log_response(parsed_param_response, 'PARAM AGENT')
                args = self.extract_params(parsed_param_response)

                self.gradio_output = param_response
                yield

                try:
                    res = tool.call(**args)
                    called = True
                except Exception as e:
                    (
                        exception_type,
                        exception_value,
                        exception_traceback
                    ) = sys.exc_info()

                    error_msg = f'Exception Type: {exception_type}'
                    error_msg += f'Exception Value: {exception_value}'

                    if 'query' in parsed_param_response['Parameter_0']:
                        if '+' in parsed_param_response['Parameter_0'][1]:
                            error_msg += sp.NO_PYTHON

                    step = {
                        'status': 'error',
                        'message': error_msg
                    }
                    completed_steps.append(step)
                    self.all_steps.append(step)
                    self.log.add(
                        'PARAM AGENT', f'Error Message: {error_msg}')
                    self.gradio_output = error_msg
                    yield

                    break

                if called:

                    # Variable name and description from function call
                    return_name = parsed_param_response[
                        'Returned'].replace(' ', '').rstrip()
                    if return_name[0] == '"' or return_name[0] == "'":
                        return_name = return_name[1:-1]
                    description = parsed_param_response['Description']

                    # Add variable to variables
                    return_var = Variable(
                        name=return_name,
                        value=res,
                        description=description,
                        dynamic=True)
                    self.variables[return_name] = return_var

                    message = (
                        f'Tool {tool_name} successfully called, ' +
                        f'Variable {return_name} saved.\n'
                    )

                    self.gradio_output = message
                    yield

                    step = {
                        'status': 'success',
                        'message': message
                    }
                    completed_steps.append(step)
                    self.all_steps.append(step)

                    message += str(res) + '\n'
                    self.log.add('SYSTEM', message)


            self.log.add('', '\n' + '*' * 60 +  '\n')

    def chat(self) -> None:
        """
        Function to give a command to interact with the LLM

        Args:
            None
        Returns:
            None
        """

        time_start = time.time()

        # Track the number of LLM calls throughout the interaction
        self.llm_calls = 0

        def get_command():
            """ Helper function to get a command from the user """
            print('\nAsk a question')
            command = ''
            while command == '':
                command = input('> ')
            if command == 'exit':
                sys.exit(1)
            return command

        while True:

            command = get_command()
            generator = self.interact(command)

            for _ in generator:
                pass

        completed_steps = []
        thoughts = ''

        while True:
            command = get_command()
            self.log.add('SYSTEM', f'User Command: {command}')

            while True:

                if self.toggle:
                    self.toggle_variables(command, thoughts)

                res = self.choose_tool(command, completed_steps)

                tool, tool_response, parsed_tool_response = res
                tool_name = parsed_tool_response['Tool']

                thoughts = str(parsed_tool_response['Thoughts'])
                if tool_name == 'execute_pandas_query':
                    thoughts += f' {sp.NO_PYTHON}'
                completed_step = (
                    'Thoughts: ' + thoughts + '\n' +
                    'Tool: ' + str(parsed_tool_response['Tool']) + '\n'
                )

                step = {
                    'status': 'success',
                    'message': completed_step
                }
                completed_steps.append(step)
                self.all_steps.append(step)

                if parsed_tool_response['Tool'] in [
                        'system_exit', 'extract_answer']:

                    # Choose parameters for the Tool
                    param_prompt = self.build_param_prompt(
                        command,
                        parsed_tool_response,
                        completed_steps)
                    param_response = self.execute(param_prompt).split('\n\n')
                    param_response = param_response[0]
                    parsed_param_response = autils.parse_response(
                        param_response)
                    self.log_response(parsed_param_response, 'PARAM AGENT')

                    args = self.extract_params(parsed_param_response)

                    if 'Parameter_0' not in parsed_param_response:

                        error_msg = 'You must generate a Parameter'
                        step = {
                            'status': 'error',
                            'message': error_msg
                        }
                        completed_steps.append(step)
                        self.all_steps.append(step)
                        self.log.add('SYSTEM', error_msg)
                        continue

                    # This is the variable that contains the answer
                    variable_name = parsed_param_response['Parameter_0'][1]

                    # Could happen if the Parameter Agent used a Python
                    # expression as the data to pass to extract_answer
                    if variable_name not in self.variables:

                        error_msg = (
                            'The Variable you have chosen '
                            f'`{variable_name}` is not an available Variable'
                        )
                        print(error_msg)
                        step = {
                            'status': 'error',
                            'message': error_msg
                        }
                        completed_steps.append(step)
                        self.all_steps.append(step)

                        self.log.add('SYSTEM', error_msg)
                        continue

                    answer = self.extract_answer(
                        command, thoughts, data=self.variables[variable_name])

                    parsed_answer = answer
                    if 'Question 1:' in answer:
                        parsed_answer = answer.split('Question 1:')[0].rstrip()
                    answer_step = f'Previous question: {command}\n'
                    answer_step += f'Previous answer: {parsed_answer}'

                    completed_steps = [
                        {
                            'status': 'success',
                             'message': answer_step
                        }
                    ]

                    answer_step += str(answer) + '\n'
                    self.log.add('SYSTEM', answer_step)

                    break

                called = False
                while not called:

                    # Choose parameters for the Tool
                    param_prompt = self.build_param_prompt(
                        command,
                        parsed_tool_response,
                        completed_steps)
                    param_response = self.execute(param_prompt).split('\n\n')
                    param_response = param_response[0]

                    parsed_param_response = autils.parse_response(
                        param_response)
                    self.log_response(parsed_param_response, 'PARAM AGENT')
                    args = self.extract_params(parsed_param_response)

                    try:
                        res = tool.call(**args)
                        called = True
                    except Exception as e:
                        (
                            exception_type,
                            exception_value,
                            exception_traceback
                        ) = sys.exc_info()

                        error_msg = f'Exception Type: {exception_type}'
                        error_msg += f'Exception Value: {exception_value}'

                        if 'query' in parsed_param_response['Parameter_0']:
                            if '+' in parsed_param_response['Parameter_0'][1]:
                                error_msg += sp.NO_PYTHON

                        step = {
                            'status': 'error',
                            'message': error_msg
                        }
                        completed_steps.append(step)
                        self.all_steps.append(step)
                        self.log.add(
                            'PARAM AGENT', f'Error Message: {error_msg}')
                        break

                    if called:

                        # Variable name and description from function call
                        return_name = parsed_param_response[
                            'Returned'].replace(' ', '').rstrip()
                        if return_name[0] == '"' or return_name[0] == "'":
                            return_name = return_name[1:-1]
                        description = parsed_param_response['Description']

                        # Add variable to variables
                        return_var = Variable(
                            name=return_name,
                            value=res,
                            description=description,
                            dynamic=True)
                        self.variables[return_name] = return_var

                        message = (
                            f'Tool {tool_name} successfully called, ' +
                            f'Variable {return_name} saved.\n'
                        )
                        step = {
                            'status': 'success',
                            'message': message
                        }
                        completed_steps.append(step)
                        self.all_steps.append(step)

                        message += str(res) + '\n'
                        self.log.add('SYSTEM', message)

                self.log.add('', '\n' + '*' * 60 +  '\n')

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
            self.log.save()
            self.save_summary(command)

    def substitute_variable_values_kwargs(self, args):
        """
        Parses the generated args to replace references of Variables in
        **kwargs with the Variable's value. If no Variable exists, then a value
        of None is used.

        Args:
        - args (dict): The dictionary of arguments to be parsed.

        Returns:
        - dict: A new dictionary with processed values based on the provided
          args.
        """

        new_args = {}
        for key, val in args.items():

            if key == 'kwargs':
                val = val.replace('{', '').replace('}', '').replace(
                    "'", '').split(',')
                for var_name in val:

                    # Once in a while the kwargs dictionary will have a
                    # different key name other than the variable
                    var_name1, var_name2 = var_name.split(':')
                    var_name1 = var_name1.strip()
                    var_name2 = var_name2.strip()
                    if var_name1 != var_name2:
                        if var_name1 in self.variables:
                            new_args[
                                var_name1] = self.variables[var_name1].value
                        elif var_name2 in self.variables:
                            new_args[
                                var_name2] = self.variables[var_name2].value
                    elif var_name1 in self.variables:
                        new_args[
                            var_name1] = self.variables[var_name1].value
                    else:
                        new_args[var_name1] = None

            else:
                new_args[key] = val
        return new_args

    def extract_params(
            self,
            parsed_data: Dict[str, Union[str, List[str]]]) -> Tuple[
                Tool, Dict[str, Any]]:
        """
        Extract parameters from LLM response

        Args:
        parsed_data (Dict): A dictionary containing the parsed data from LLM.

        Returns:
            tool (Tool): The tool extracted from available tools based on
            'Tool' key in parsed_data.
            args (Dict): A dictionary of arguments to be passed to the
            function.
        """

        param_prefix = 'Parameter_'

        # Create a dictionary of arguments to be passed to the function
        args = {}
        for key, value in parsed_data.items():
            if key.startswith(param_prefix):

                # Value does not contain name | value | type | val/ref
                if len(value) != 4:
                    continue

                param_name = value[0]
                param_value = value[1]
                param_type = value[2].replace(' ', '')
                param_vr = value[3]

                # The parameter is a reference to a Variable in self.variables
                if 'reference' in param_vr:
                    # Strip out any quotes that might be at the beginning/end
                    param_value = autils.strip_quotes(param_value)

                    if param_value in self.variables:
                        param_value = self.variables[param_value].value

                # Parameter is a string but not a SQL query
                elif 'str' in param_type and param_name != 'query':
                    param_value = autils.strip_quotes(param_value)

                # Cast the values if needed
                if 'value' in param_vr:
                    if param_type == 'int':
                        param_value = int(param_value)
                    elif param_type == 'float':
                        param_value = float(param_value)
                    elif param_type == 'bool':
                        if param_value.lower() == 'true':
                            param_value = True
                        elif param_value.lower() == 'false':
                            param_value = False
                        else:
                            param_value=None

                args[param_name.replace(' ', '')] = param_value

        if 'kwargs' in args:
            args = self.substitute_variable_values_kwargs(args)

        return args

    def extract_answer(self, question: str, thoughts: str, data) -> str:
        """
        Extracts an answer to a given question.

        Args:
            question (str): The question for which the answer is to be
            extracted.
            thoughts (str): The thoughts from the LLM

        Returns:
            extracted_answer (str): The extracted answer to the question.
        """
        prompt = Prompt()
        prompt.add_base_prompt(sp.ANSWER_QUESTION_PROMPT)
        prompt.add_command(question)
        prompt.add_command(thoughts)
        prompt.add_variable(data, truncate=False)
        return self.execute(prompt.generate_prompt(), True, True)

    def execute(
            self,
            prompt_str: str,
            print_prompt: bool=True,
            print_response: bool=True) -> str:
        """
        Calls the LLM, updates the running cost, adds the prompt and response
        to the log, and prints the prompt/response.

        Args:
            prompt (str): The prompt for the LLM
            print_prompt (bool): If True, prints the prompt
            print_response (bool): If True, prints the response

        Returns:
            response (str): The response from the LLM.
        """
        self.llm_calls += 1
        if print_prompt or print_response:
            print('Thinking...')
        self.input_cost += autils.calculate_prompt_cost(
            self.model_info, prompt_str)['input']
        if print_prompt:
            print('#' * 60 + ' PROMPT BEGIN ' + '#' * 60)
            print(prompt_str)
            print('#' * 60 + ' SYSTEM PROMPT END ' + '#' * 60)
            print('\n')
        response = oaiapi.chat_completion(
            prompt=prompt_str, model=self.model_info.name)
        self.output_cost += autils.calculate_prompt_cost(
            self.model_info, response)['output']
        if print_response:
            print('#' * 60 + ' RESPONSE BEGIN ' + '#' * 60)
            print(response)
            print('#' * 60 + ' RESPONSE END ' + '#' * 60)
            print('\n')
        return response

    def save_summary(self, command) -> None:
        """
        Generates a short summary based on the steps taken.

        Args:
            command (str): The initial command from the user

        Returns:
            None
        """
        print('\nGenerating a summary of the interaction...')
        completed_steps_str = []
        for step_dict in self.all_steps:
            status = step_dict['status']
            message = step_dict['message']
            step_str = f'Status: {status}\n'
            step_str += f'Message: {message}\n'
            completed_steps_str.append(step_str)

        steps_summary = autils.summarize_completed_steps(
            command, completed_steps_str)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'log_{timestamp}-summary.txt'
        filepath = opj(self.save_dir, filename)
        with open(filepath, 'w') as file:
            file.write(steps_summary)
        print('Summary saved to', filepath)

