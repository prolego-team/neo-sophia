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

    # two functions and 3 parameters between them
    lines = prompt.split('\n')
    assert len([x for x in lines if x.startswith('name:')]) == 2
    assert len([x for x in lines if x.startswith('description:')]) == 2
    assert len([x for x in lines if x.startswith('parameter:')]) == 3


def test_parse_dispatch_response():
    """Test dispatch response parsing"""

    functions = {
        'combine': dp.FunctionDesc(
            'combine parameters somehow',
            params={
                'a': dp.ParamDesc('First parameter', int, True),
                'b': dp.ParamDesc('Second parameter', float, True),
                'c': dp.ParamDesc('Third parameter', str, True),
                'd': dp.ParamDesc('Fourth parameter', str, True)
            }
        )
    }

    # correct format
    name, params = dp.parse_dispatch_response(
        response=(
            'blah blah baloney\n' +
            'FUNCTION: combine\n' +
            'PARAMETER: a=5\n' +
            'PARAMETER: b=6\n' +
            'PARAMETER: c="Hello, world!\nMultiple\nlines\nof text."\n' +
            'PARAMETER: d=Done.'
        ),
        functions=functions
    )
    assert name == 'combine'
    assert params == {'a': 5, 'b': 6.0, 'c': 'Hello, world!\nMultiple\nlines\nof text.', 'd': 'Done.'}

    # name doesn't match
    res = dp.parse_dispatch_response(
        response='FUNCTION: baloney\nPARAMETER: a=5',
        functions=functions
    )
    assert res is None

    # param names don't match
    name, params = dp.parse_dispatch_response(
        response=(
            'FUNCTION: combine\n' +
            'PARAMETER: a=5\n' +
            'PARAMETER: b=6\n' +
            'PARAMETER: e="Hello, world!"'
        ),
        functions=functions
    )
    assert name == 'combine'
    assert params == {'a': 5, 'b': 6.0}

    # param types can't be parsed
    name, params = dp.parse_dispatch_response(
        response=(
            'FUNCTION: combine\n' +
            'PARAMETER: a="Hello, world!"\n' +
            'PARAMETER: b=6\n' +
            'PARAMETER: c="Hello, world!"\n' +
            'PARAMETER: d=Done.'
        ),
        functions=functions
    )
    assert name == 'combine'
    assert params == {'b': 6.0, 'c': 'Hello, world!', 'd': 'Done.'}


def test_convert_function_descs():
    """test conver_function_descs"""

    res = dp.convert_function_descs(EXAMPLE_FUNCTIONS)

    # assert a few things

    assert len(res) == 2

    assert res[0]['name'] == 'find_document'
    assert list(res[0]['parameters']['properties'].keys()) == ['search_str', 'n_closest']
    assert res[0]['parameters']['required'] == ['search_str']

    assert res[1]['name'] == 'query_db'
    assert list(res[1]['parameters']['properties'].keys()) == ['query_str']
    assert res[1]['parameters']['required'] == ['query_str']
