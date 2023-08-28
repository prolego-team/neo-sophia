""" Class for generating a structured prompt """
import re

from functools import partial

import pandas as pd


def format_df(df):
    """
    Function that stripts out all whitespace between dataframe columns while
    preserving white space in individual cells
    """
    def get_fmt_str(x, fill):
        return '| {message: >{fill}} '.format(message=x, fill=fill-2)

    # Max character length per column
    s = df.astype(str).agg(lambda x: x.str.len()).max()
    pad = 0  # How many spaces between
    fmts = {}
    header_strs = []
    for idx, c_len in s.items():
        if isinstance(idx, tuple):
            lab_len = max([len(str(x)) for x in idx])
        else:
            lab_len = len(str(idx))

        fill = max(lab_len, c_len) + pad
        fmts[idx] = partial(get_fmt_str, fill=fill)

        # Formatting the header
        header_strs.append(get_fmt_str(idx, fill))

    # Generate the formatted DataFrame string without the header
    df_str = df.to_string(formatters=fmts, index=False, header=False)

    # Generate the header string with the | separator
    header = ''.join(header_strs) + '|'

    # Combine the header and the DataFrame string
    final_str = header + '\n' + df_str
    return re.sub(r'[ \t]*\|[ \t]*', '|', final_str)


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

    def add_base_prompt(self, prompt):
        """ This prompt always goes at the beginning """
        self.base_prompt.append(prompt + '\n')

    def add_command(self, command):
        self.commands.append(command)

    def add_example(self, example):
        self.examples.append(example)

    def add_variable(self, variable, visible=False):
        """

        """
        dots = ''
        if visible or variable.visible:
            if isinstance(variable.value, pd.DataFrame):
                value = variable.value
                if not value.empty:
                    value = value.head(10)
                    dots = '...\n'
                value = format_df(value)
            else:
                value = variable.value
            var_type = str(type(variable.value).__module__)
            var_type += '.' + str(type(variable.value).__name__)
            prompt = f'Name: {variable.name}\n'
            prompt += f'Type: {var_type}\n'
            prompt += f'Value:\n{value}\n{dots}'
            prompt += f'Description: {variable.description}\n'
            prompt += '\n'
            self.variables.append(prompt)

    def add_resource(self, resource, visible=False):
        if visible or resource.visible:
            prompt = f'Name: {resource.name}\n'
            prompt += f'Description: {resource.description}\n'
            self.resources.append(prompt)

    def add_tool(self, tool):
        self.tools.append(str(tool))

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
        if self.variables:
            prompt += _construct('VARIABLES', self.variables)
        if self.constraints:
            prompt += _construct('CONSTRAINTS', self.constraints)
        if self.examples:
            for idx, example in enumerate(self.examples):
                prompt += f'EXAMPLE {idx + 1}:\n{example}\n'
        if self.steps:
            prompt += _construct('COMPLETED STEPS', self.steps)

        prompt += tot * dash
        return prompt

