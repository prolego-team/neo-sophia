""" Class for generating a structured prompt """
import re

from typing import Dict
from functools import partial

import pandas as pd

from neosophia.agents.data_classes import Tool, Variable


def format_df(df: pd.DataFrame, delimeter: str='|') -> str:
    """
    Function that strips out all whitespace between dataframe columns and
    replaces it with the specified delimeter.

    Example usage:

        [Input]

             guid              name         dob
        0       1   Patricia Karins  1991-04-19
        1       2       Lillian Lee  1992-05-20
        2       3  Richard Crandall  1980-11-24
        3       4  Billy Harrington  1970-06-17
        4       5          Al Elzey  1940-01-01
        --

        [Output]

        |guid|name|dob|
        |1|Patricia Karins|1991-04-19|
        |2|Lillian Lee|1992-05-20|
        |3|Richard Crandall|1980-11-24|
        |4|Billy Harrington|1970-06-17|
        |5|Al Elzey|1940-01-01|

    Args:
        df (pandas.DataFrame): The input dataframe to be formatted.

    Returns:
        output (str): The formatted DataFrame string with stripped
        whitespace between columns and preserved whitespace in individual
        cells.
    """

    # Convert DataFrame to string
    output = df.to_csv(
        sep=delimeter, index=False, header=True, lineterminator='\n')

    # Add the delimeter to the beginning and end of each line
    output = '\n'.join(
        [delimeter + line + delimeter for line in output.strip().split('\n')])
    return output


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
        self.variables = []
        self.errors = []

    def add_base_prompt(self, prompt: str) -> None:
        """
        Adds a base prompt that always goes at the beginning

        Args:
            prompt (str): The base prompt to be added.

        Returns:
            None
        """
        self.base_prompt.append(prompt + '\n')

    def add_command(self, command: str) -> None:
        """
        Adds a command to the list of commands

        Args:
            command (str): The command from the user
        Returns:
            None
        """
        pre = 'USE THE LEAST AMOUNT OF TOOLS POSSIBLE\n'
        pre += 'ONLY USE A SINGLE SQL QUERY WITH THE TOOL execute_pandas_query IF POSSIBLE\n'
        self.commands.append(pre + command)

    def add_command(self, command: str) -> None:
        """
        Adds a command to the list of commands

        Args:
            command (str): The command from the user
        Returns:
            None
        """
        self.commands.append(command)

    def add_example(self, example: str) -> None:
        """
        Adds an example to the list of examples

        Args:
            example (str): The example to add
        Returns:
            None
        """
        self.examples.append(example)

    def add_variable(
            self,
            variable: Variable,
            visible: bool=False,
            truncate: bool=True) -> None:
        """
        Adds a Variable to the list of Variables. If the Variable is a pandas
        dataframe and has more than 10 rows of data, only the first 10 are
        shown, UNLESS it's a schema. If that's the case, then we show all of
        the rows

        Args:
            variable (Variable): The variable to add
            visible (bool): Whether or not the Variable should be shown
        Returns:
            None
        """
        dots = ''
        max_rows = 5
        truncated = False
        if visible or variable.visible:
            if isinstance(variable.value, pd.DataFrame):
                value = variable.value
                if not value.empty:
                    num_rows = value.shape[0]
                    if 'schema' not in variable.name and num_rows > max_rows:
                        if truncate:
                            value = value.head(max_rows)
                            num_tr = num_rows - max_rows
                            dots = f'... ({num_tr} rows truncated)\n'
                            truncated = True
                value = format_df(value)
            else:
                value = variable.value
            var_type = str(type(variable.value).__module__)
            var_type += '.' + str(type(variable.value).__name__)
            prompt = f'Name: {variable.name}\n'
            prompt += f'Type: {var_type}\n'
            prompt += f'Value:\n{value}\n{dots}'
            prompt += f'Truncated: {truncated}\n'
            prompt += f'Description: {variable.description}\n'
            prompt += '\n'
            self.variables.append(prompt)

    def add_tool(self, tool: Tool) -> None:
        """
        Adds a Tool to the list of Tools

        Args:
            tool (Tool): The Tool to add
        Returns:
            None
        """
        self.tools.append(tool.to_string())

    def add_constraint(self, constraint: str) -> None:
        """
        Adds a constraint to the list of constraints

        Args:
            constraint (str): The constraint to add
        Returns:
            None
        """
        self.constraints.append(constraint)

    def add_completed_step(self, step: Dict[str, str]) -> None:
        """
        Adds a step to the list of completed_steps

        Args:
            step (Dict[str, str]): The step to add with its type.
            e.g.,
            `step = {'status': 'success', message: <step details>}`
            `step = {'status': 'error', message: <error message>}`
        Returns:
            None
        """
        self.steps.append(step)

    def add_error(self, error: str) -> None:
        """
        Adds an error to the list of errors

        Args:
            error (str): The error to add
        Returns:
            None
        """
        self.errors.append(error)

    def generate_prompt(self, tot: int=80) -> str:
        """
        Organizes the base prompt, Variables, Tools, constraints,
        completed steps, and commands to generate a string prompt for the LLM.

        args:
            tot (int): Number of '-' to use for the different sections
        Returns:
            prompt (str): The generated prompt
        """
        user_prompt = ''
        dash = '-'

        def _get_dash(text):
            """ Helper function to calculate the number of dashes to use """
            n = (tot - len(text)) // 2
            extra = '' if len(text) % 2 == 0 else dash
            return dash * n + extra + text + dash * n

        def _construct(text, items):
            """ Constructs a section with correct dashes and newlines """
            items = '\n'.join(items) + '\n\n'
            prompt = _get_dash(text) + '\n'
            prompt += items
            return prompt

        if self.tools:
            user_prompt += _construct('TOOLS', self.tools)
        if self.resources:
            user_prompt += _construct('DATA RESOURCES', self.resources)
        if self.variables:
            user_prompt += _construct('VARIABLES', self.variables)
        if self.constraints:
            user_prompt += _construct('CONSTRAINTS', self.constraints)
        if self.examples:
            user_prompt += _get_dash('EXAMPLES') + '\n'
            for idx, example in enumerate(self.examples):
                user_prompt += f'EXAMPLE {idx + 1}:{example}\n\n'
        if self.steps:
            steps = []
            for step in self.steps:
                status = step['status']
                message = step['message']
                steps.append(
                    f'Step Status: {status}\nMessage: {message}\n')
            user_prompt += _construct(
                'COMPLETED STEPS', [x + '\n--\n' for x in steps])
        if self.commands:
            user_prompt += _construct('COMMANDS', self.commands)
        if self.errors:
            user_prompt += _construct('ERRORS', self.errors)
        if self.base_prompt:
            user_prompt += _construct('SYSTEM PROMPT', self.base_prompt)

        user_prompt += tot * dash

        return user_prompt
