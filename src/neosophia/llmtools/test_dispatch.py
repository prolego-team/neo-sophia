"""
Tests for dispatch functionality.
"""

from neosophia.llmtools import dispatch as dp


def test_dataclasses():
    """Test function dataclasses."""

    functions = {
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
        )
    }

    prompt = dp.make_prompt(
        question='Find the document that describes the process for second mortgages.',
        functions=functions
    )

    print(prompt)

    # TODO: assertions
