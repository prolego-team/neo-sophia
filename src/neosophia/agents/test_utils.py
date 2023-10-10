"""

"""
import os
import unittest

from neosophia.agents.utils import *
from neosophia.agents.data_classes import GPTModelInfo, GPT_MODELS

MODEL = 'gpt-4-0613'

# Sample configuration of what might be loaded from config.yaml
TEST_CONFIG = {
    'Agent': {
        'name': 'MyAgent',
        'workspace_dir': '.temp_dir',
        'resources_filename': 'resources.yaml',
        'tools_filename': 'tools.yaml'
    },
    'Tools': [
        {
            'module': 'neosophia.agents.tools',
            'functions': [
                'get_min_values',
                'get_max_values'
            ]
        }
    ]
}


def test_calculate_prompt_cost():

    prompt = """Amet etincidunt consectetur quisquam ut eius. Dolore quiquia ut
    amet dolor sed magnam. Consectetur amet aliquam dolor quaerat labore est
    sit. Quaerat dolorem quiquia etincidunt. Sed sed etincidunt voluptatem
    numquam etincidunt neque.  Consectetur numquam neque quiquia voluptatem.
    Magnam aliquam neque magnam sed eius quisquam. Dolore quisquam labore
    sed velit tempora amet quaerat. Porro quaerat ut quisquam neque adipisci
    modi. Sed ut magnam ut quiquia neque.  Dolorem aliquam eius consectetur
    neque. Tempora adipisci labore voluptatem quisquam voluptatem dolor.
    Adipisci ut non dolor. Quisquam sit velit labore numquam consectetur
    quisquam quisquam. Aliquam adipisci eius non aliquam sed.  Neque eius
    consectetur quaerat eius. Quaerat eius porro sed velit dolore eius.  Sit
    est ipsum dolore. Neque consectetur adipisci adipisci."""

    expected_cost = {
        'input': 0.007259999999999999,
        'output': 0.014519999999999998
    }
    model_info = GPT_MODELS[MODEL]
    cost = calculate_prompt_cost(model_info, prompt)
    assert cost == expected_cost


def test_create_workspace_dir():
    # Remove directory if it exists
    workspace_dir = TEST_CONFIG['Agent']['workspace_dir']
    if os.path.exists(workspace_dir):
        os.rmdir(workspace_dir)
    create_workspace_dir(TEST_CONFIG)
    x = os.path.exists(workspace_dir)
    assert x
    os.rmdir(workspace_dir)


def test_build_function_dict_from_module():
    import neosophia.agents.tools as module

    function_names = ['get_max_values']

    expected_function_text = (
        "\n\ndef get_max_values(df: pd.DataFrame) -> pd.Series:\n"
        "    '\\n    Get the maximum value of each column in the DataFrame.\\n"
        "\\n    Parameters:\\n    - df (pd.DataFrame): The input DataFrame.\\n"
        "\\n    Returns:\\n    - pd.Series: A series containing the maximum value of each column.\\n"
        "    '\n    return df.max()\n"
    )
    function_dict = build_function_dict_from_module(
        module, function_names)

    expected_function_dict = {
        'get_max_values': (module.get_max_values, expected_function_text)
    }
    assert function_dict == expected_function_dict


def test_parse_response():

    response = """Parameter_0: query | 'SELECT COUNT(*) AS total_customers FROM customers_data' | str | 'value'
Parameter_1: kwargs | {'customers_data': customers_data} | Dict[(str, pd.DataFrame)] | 'reference'
Returned: total_customers
Description: The total number of customers in database SynthBank"""

    expected_parsed_response = {
        'Parameter_0': [
            'query',
            "'SELECT COUNT(*) AS total_customers FROM customers_data'",
            'str',
            "'value'"],
        'Parameter_1': [
            'kwargs',
            "{'customers_data': customers_data}", 'Dict[(str, pd.DataFrame)]',
            "'reference'"
        ],
        'Returned': 'total_customers',
        'Description': 'The total number of customers in database SynthBank'
    }

    # TODO - add more complex parameters here
    response = (
        "Thoughts: Random thoughts from the LLM\n"
        "Tool: execute_query\n"
        "Parameter_0: conn | Baloney_conn | sqlite3.Connection | reference\n"
        "Parameter_1: query | 'SELECT * FROM baloney' | str | value\n"
        "Returned: lots_of_baloney\n"
        "Description: DataFrame containing a TON of baloney\n"
    )
    expected_parsed_response = {
        'Thoughts': 'Random thoughts from the LLM',
        'Tool': 'execute_query',
        'Parameter_0': [
            'conn', 'Baloney_conn', 'sqlite3.Connection', 'reference'
        ],
        'Parameter_1': ['query', "'SELECT * FROM baloney'", 'str', 'value'],
        'Returned': 'lots_of_baloney',
        'Description': 'DataFrame containing a TON of baloney'
    }

    #parsed_response = parse_response(response)

    #assert expected_parsed_response, parsed_response)

    response = (
        "Thoughts: Use the 'execute_query' tool to get information.\n\n"
        "Tool: execute_query\n"
        "Parameter_0: conn|Transactions_conn|sqlite3.Connection|reference\n"
        "Parameter_1: query|'SELECT MAX(a) FROM transactions'|str|values\n"
        "Returned: max_trans_values\n"
        "Description: DataFrame containing the transaction ID\n"
    )
    parsed_response = parse_response(response)

    response = 'Apologies, but no command was provided'


def test_count_tokens():
    num_tokens = count_tokens('sample string to count tokens', MODEL)
    assert num_tokens == 5


def test_remove_yaml_special_chars():

    text = (
        'name: John, age: 25! city: #New York > country: USA\n'
        'person: {name: Alice, details: {age: 30}}\n'
        'fruits: [apple, banana, orange] & colors: {red, blue}\n'
        'description: "This is a *description with : & ,"'
    )

    expected_out = (
        'name John age 25 city New York  country USA\n'
        'person name Alice details age 30\n'
        'fruits apple banana orange  colors red blue\n'
        'description "This is a description with   "'
    )

    out = remove_yaml_special_chars(text)

    assert out == expected_out


def test_process_for_yaml():

    name = 'Database'
    description = 'This is the database description'

    expected_out = (
        '- name: Database\n'
        '  description: This is the database description'
    )

    out = process_for_yaml(name, description)

    assert out == expected_out

