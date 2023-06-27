"""
Tests for OpenAI API.
"""

from unittest.mock import patch

from neosophia.llmtools import openaiapi


@patch('neosophia.llmtools.openaiapi.embeddings')
def test_embeddings_tensor(embeddings_mock):
    """test `embeddings_tensor` functions"""
    n_texts = 3

    # OpenAI API returns a data struction which contains an embedding
    # for each input
    example_data = {
        'data': [
            {
                'embedding': [0.0] * openaiapi.EMBEDDING_DIM_DEFAULT
            }
        ] * n_texts

    }
    embeddings_mock.return_value = example_data
    texts = ['baloney'] * n_texts

    res = openaiapi.embeddings_tensor(texts)
    assert res.shape == (n_texts, openaiapi.EMBEDDING_DIM_DEFAULT)
