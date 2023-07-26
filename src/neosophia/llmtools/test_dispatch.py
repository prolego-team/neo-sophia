"""
Tests for dispatch functionality.
"""

from neosophia.llmtools import dispatch as dp


EXAMPLE_FUNCTIONS = {
    'find_document': dp.FunctionDesc(
        description='Search a vector database. Returns the ids and text of one or more matching documents',
        params={
            'search_str': dp.ParamDesc(
                description='Search string',
                typ=str,
                required=True
            ),
            'n_closest': dp.ParamDesc(
                description='Number of closest matching documents to return.',
                typ=int,
                required=False
            )
        }
    ),
    'query_db': dp.FunctionDesc(
        description=(
            'Run a query on a SQLite database. ' +
            'Contains a Customers table with Name, Address, Checking, Savings, and RothIRA columns'
        ),
        params={
            'query_str': dp.ParamDesc(
                description='query string',
                typ=str,
                required=True
            )
        }
    )
}


def test_dataclasses():
    """Test function dataclasses and conversion to prompts."""

    prompt = dp.dispatch_prompt(
        question='Find up to three documents that describes the process for second mortgages.',
        functions=EXAMPLE_FUNCTIONS
    )

    print(prompt)

    # TODO: assertions
