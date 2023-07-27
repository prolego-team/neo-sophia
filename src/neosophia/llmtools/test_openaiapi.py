"""
Tests for OpenAI API.
"""

from unittest.mock import patch

import pytest

from neosophia.llmtools import openaiapi

@pytest.fixture
def openai_chat_response():
    def _chat_output(content, role):
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": content,
                        "role": role
                    }
                }
            ],
            "created": 1689623190,
            "id": "chatcmpl-xyz",
            "model": "gpt-3.5-turbo-0613-mock",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 99,
                "prompt_tokens": 99,
                "total_tokens": 99
            }
        }

    return _chat_output


@patch('neosophia.llmtools.openaiapi.embeddings')
def test_embeddings_tensor(embeddings_mock):
    """test `embeddings_tensor` and `extract_embeddings` functions"""
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


def test_messages(openai_chat_response):
    message = openaiapi.Message('system', 'You are a helpful assistant.')
    assert message.is_valid()
    assert message.as_dict()=={
        'role': 'system',
        'content': 'You are a helpful assistant.'
    }

    target_output_content = 'This is output from the LLM.'
    target_output_role = 'assistant'
    response = openai_chat_response(
        target_output_content,
        target_output_role
    )
    message = openaiapi.Message.from_api_response(response)
    assert message.is_valid()
    assert message==openaiapi.Message(
        role=target_output_role,
        content=target_output_content,
        name=None
    )

    message = openaiapi.Message.from_function_call('get_weather', 72)
    assert message.is_valid()
    assert message==openaiapi.Message(
        role='function',
        content='72',
        name='get_weather'
    )


@patch('neosophia.llmtools.openaiapi.oai.ChatCompletion.create')
def test_chat_completion(chat_mock, openai_chat_response):
    target_output_content = 'Who is there?'
    target_output_role = 'assistant'
    chat_mock.return_value = openai_chat_response(
        target_output_content,
        target_output_role
    )
    model = openaiapi.start_chat('test_model')
    messages = [
        openaiapi.Message('system', 'You are a helpful assistant.'),
        openaiapi.Message('user', 'Knock knock.')
    ]
    response = model(messages)
    assert response.content==target_output_content
    assert response.role==target_output_role
