"""

"""
import os
import unittest

from neosophia.agents.utils import *
from neosophia.agents.data_classes import Resource

MODEL = 'gpt-4-0613'

# Sample configuration of what might be loaded from config.yaml
TEST_CONFIG = {
    'Agent': {
        'name': 'MyAgent',
        'workspace_dir': '.temp_dir',
        'resources_filename': 'resources.yaml',
        'tools_filename': 'tools.yaml'
    },
    'Resources': {
        'SQLite': [
            {
                'name': 'SynthBank',
                'path': 'data/synthbank.db',
                'description': None
            },
        ]
    },
    'Tools': [
        {
            'module': 'neosophia.db.sqlite_utils',
            'functions': [
                'execute_query'
            ]
        },
        {
            'module': 'neosophia.agents.tools',
            'functions': [
                'dataframe_intersection'
            ]
        }
    ]
}

WORKSPACE_RESOURCES = {
    'SynthBank': {
        'name': 'SynthBank',
        'path': 'data/synthbank.db',
        'description': 'The database description'
    }
}


class TestUtils(unittest.TestCase):

    def test_setup_sqlite_resources(self):
        x = setup_sqlite_resources(
            TEST_CONFIG['Resources']['SQLite'],
            TEST_CONFIG['Agent']['workspace_dir'],
            TEST_CONFIG['Agent']['resources_filename'],
            WORKSPACE_RESOURCES
        )
        y = {
            'SynthBank': Resource(
                name='SynthBank',
                path='data/synthbank.db',
                description='The database description')
        }
        self.assertEqual(x, y)

    def test_create_workspace_dir(self):
        # Remove directory if it exists
        workspace_dir = TEST_CONFIG['Agent']['workspace_dir']
        if os.path.exists(workspace_dir):
            os.rmdir(workspace_dir)
        create_workspace_dir(TEST_CONFIG)
        x = os.path.exists(workspace_dir)
        self.assertEqual(x, True)
        os.rmdir(workspace_dir)

    def test_build_function_dict_from_module(self):
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
        self.assertEqual(function_dict, expected_function_dict)

    def test_parse_response(self):

        # TODO - add more complex parameters here
        response = (
            "Thoughts: Random thoughts from the LLM\n"
            "Resource: Baloney\n"
            "Tool: execute_query\n"
            "Parameter_0: conn | Baloney_conn | sqlite3.Connection | reference\n"
            "Parameter_1: query | 'SELECT * FROM baloney' | str | value\n"
            "Returned: lots_of_baloney\n"
            "Description: DataFrame containing a TON of baloney\n"
        )
        expected_parsed_response = {
            'Thoughts': 'Random thoughts from the LLM',
            'Resource': 'Baloney',
            'Tool': 'execute_query',
            'Parameter_0': [
                'conn', 'Baloney_conn', 'sqlite3.Connection', 'reference'
            ],
            'Parameter_1': ['query', "'SELECT * FROM baloney'", 'str', 'value'],
            'Returned': 'lots_of_baloney',
            'Description': 'DataFrame containing a TON of baloney'
        }

        parsed_response = parse_response(response)

        self.assertEqual(expected_parsed_response, parsed_response)

    def test_count_tokens(self):
        num_tokens = count_tokens('sample string to count tokens', MODEL)
        self.assertEqual(num_tokens, 5)


if __name__ == '__main__':
    unittest.main()
