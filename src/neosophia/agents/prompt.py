"""
"""
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
        self.base_prompt.append(prompt + '\n')

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

