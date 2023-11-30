"""
"""
import re
import pickle

import lorem
import pandas as pd

import neosophia.agents.utils as autils

from neosophia.agents.agent import Agent
from neosophia.agents.data_classes import Tool, Variable


def _get_agent():
    tools = {
        'dummy_tool': Tool(
            name='dummy_tool',
            function_str='',
            description='Does nothing',
            call=lambda x: None
        )
    }
    variables = {
        'dummy_variable': Variable(
            name='dummy_variable',
            value=100,
            description='Dummy variable',
            visible=True
        )
    }
    return Agent(
        workspace_dir='.temp',
        tool_bp=None,
        param_bp=None,
        tools=tools,
        variables=variables,
        toggle=False)


def test_check_prompt():
    """ """

    agent = _get_agent()
    prompt1 = lorem.text()
    out1 = agent.check_prompt(prompt1)

    prompt2 = lorem.text() * 100
    out2 = agent.check_prompt(prompt2)

    assert out1
    assert not out2


def test_build_tool_prompt():
    """ """

    agent = _get_agent()
    base_prompt = 'Build tool base prompt'
    command = 'This is the user command'

    step1 = {
        'status': 'success',
        'message': 'Message 1'
    }
    step2 = {
        'status': 'success',
        'message': 'Message 2'
    }
    completed_steps = [step1, step2]

    tool_prompt = agent.build_tool_prompt(base_prompt, command, completed_steps)
    expected_tool_prompt = """--------------------------------------TOOLS-------------------------------------

Tool Name: dummy_tool
Description: Does nothing


Tool Name: extract_answer
Description: Tool Name: extract_answer
Description: This function extracts an answer given a question and a dataframe containing the answer.
params:
  data:
    description: A DataFrame or Series that contains the answer
    required: true


Tool Name: system_exit
Description: Tool to exit the program


------------------------------------VARIABLES-----------------------------------
Name: dummy_variable
Type: builtins.int
Value:
100
Truncated: False
Description: Dummy variable



---------------------------------COMPLETED STEPS--------------------------------
Step Status: success
Message: Message 1

--

Step Status: success
Message: Message 2

--


------------------------------------COMMANDS------------------------------------
This is the user command

----------------------------------SYSTEM PROMPT---------------------------------
Build tool base prompt


--------------------------------------------------------------------------------"""
    assert tool_prompt == expected_tool_prompt


def test_build_param_prompt():
    agent = _get_agent()
    agent.param_bp = 'Param base prompt'
    parsed_response = {
        'Thoughts': 'Thoughts from the agent',
        'Tool': 'dummy_tool'
    }
    command = 'This is the user command'

    step1 = {
        'status': 'success',
        'message': 'Message 1'
    }
    step2 = {
        'status': 'success',
        'message': 'Message 2'
    }
    completed_steps = [step1, step2]

    prompt = agent.build_param_prompt(command, parsed_response, completed_steps)

    expected_param_prompt = """--------------------------------------TOOLS-------------------------------------

Tool Name: dummy_tool
Description: Does nothing


------------------------------------VARIABLES-----------------------------------
Name: dummy_variable
Type: builtins.int
Value:
100
Truncated: False
Description: Dummy variable



-----------------------------------CONSTRAINTS----------------------------------
Do not engage in conversation or provide an explanation. Simply provide an answer.
Do not provide Thoughts
Only provide one set of Parameters at a time
Do not say what tool you are using
Only generate Parameters, the Returned name, and a Description as defined in the template
Do not generate an SQL query that contains a Python expression. You must use exact values in the SQL queries that you generate

---------------------------------COMPLETED STEPS--------------------------------
Step Status: success
Message: Message 1

--

Step Status: success
Message: Message 2

--


------------------------------------COMMANDS------------------------------------
This is the user command
Thoughts from the agent

----------------------------------SYSTEM PROMPT---------------------------------
Param base prompt


--------------------------------------------------------------------------------"""
    assert prompt == expected_param_prompt


def test_extract_params():

    parsed_response = {
        'Parameter_0': [
            'query',
            "'SELECT * AS dummy_returned FROM dummy_variable'",
            'str',
            "'value'"],
        'Parameter_1': [
            'kwargs',
            "{'dummy_variable': dummy_variable}", 'Dict[(str, pd.DataFrame)]',
            "'reference'"
        ],
        'Returned': 'dummy_returned',
        'Description': 'Description for dummy variable'
    }

    agent = _get_agent()
    expected_args = {
        'query': "'SELECT * AS dummy_returned FROM dummy_variable'",
        'dummy_variable': 100
    }

    args = agent.extract_params(parsed_response)

    assert args == expected_args


def test_substitute_variable_values_kwargs():

    agent = _get_agent()

    # Example 1 - expected generated kwargs
    args = {
        'query': "'SELECT * FROM dummy_variable'",
        'kwargs': "{'dummy_variable': dummy_variable}"
    }
    parsed_args = agent.substitute_variable_values_kwargs(args)
    expected_args = {
        'query': "'SELECT * FROM dummy_variable'",
        'dummy_variable': 100
    }
    assert parsed_args == expected_args

    # Example 2 - key in kwargs is different
    args = {
        'query': "'SELECT * FROM dummy_variable'",
        'kwargs': "{'dummy': dummy_variable}"
    }
    parsed_args = agent.substitute_variable_values_kwargs(args)
    expected_args = {
        'query': "'SELECT * FROM dummy_variable'",
        'dummy_variable': 100
    }
    assert parsed_args == expected_args

    # Example 2 - val in kwargs is different
    args = {
        'query': "'SELECT * FROM dummy_variable'",
        'kwargs': "{'dummy_variable': dummy}"
    }
    parsed_args = agent.substitute_variable_values_kwargs(args)
    expected_args = {
        'query': "'SELECT * FROM dummy_variable'",
        'dummy_variable': 100
    }
    assert parsed_args == expected_args

