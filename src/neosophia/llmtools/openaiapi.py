"""
Wrappers for OpenAI API.
"""

from typing import Any, List, Callable, Optional
import os
import sys
from dataclasses import dataclass

import openai as oai
import torch


EMBEDDING_DIM_DEFAULT = 1536
EMBEDDING_MODEL_DEFAULT = 'text-embedding-ada-002'


def set_api_key(api_key: str) -> None:
    """set the API key"""
    oai.api_key = api_key


def load_api_key(file_path: str) -> str:
    """load the API key from a text file"""
    if not os.path.isfile(file_path):
        print(f'OpenAI API key file `{file_path}` not found!')
        sys.exit()
    with open(file_path, 'r') as txt_file:
        res = txt_file.read().rstrip()
    if not res:
        print(f'Key file `{file_path}` empty!')
        sys.exit()

    return res


def get_models_list() -> List:
    """Return a list of available models."""
    model_obj= oai.Model.list()
    return [model.id for model in model_obj.data]


def embeddings(texts: List[str]) -> Any:
    """get embeddings, specifying the default model"""
    return oai.Embedding.create(
        input=texts,
        model=EMBEDDING_MODEL_DEFAULT
    )


def chat_completion(prompt: str, model: str) -> str:
    """simple chat completion"""
    res = oai.ChatCompletion.create(
        model=model,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
    )
    return res['choices'][0]['message']['content']


def extract_embeddings(data: Any) -> torch.Tensor:
    """extract embedings from an API response"""
    embs = [
        torch.tensor(x['embedding'])
        for x in data['data']]
    return torch.stack(embs, dim=0)


def embeddings_tensor(texts: List[str]) -> torch.Tensor:
    """get a tensor of embeddings from a list of strings"""
    embs = embeddings(texts)
    res = extract_embeddings(embs)
    assert res.shape == (len(texts), EMBEDDING_DIM_DEFAULT)
    return res


@dataclass
class Message:
    """Simple data structure for working with the messages that are passed back
    and forth to the OpenAI chat APIs.

    The role and content attributes are required when preparing a message to pass
    to OpenAI.  The response message may contain either content or function_call
    in addition to role.
    """
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[dict] = None

    def is_valid(self) -> bool:
        """Make sure this is a valid OpenAI message."""
        valid = True
        valid &= self.role in ['system', 'user', 'assistant', 'function']
        valid &= (len(self.content)>0) or (self.function_call is not None)
        return valid

    def as_dict(self) -> dict:
        """Because it's much, much faster than built in dataclasses.asdict."""
        return {key:value for key,value in vars(self).items() if value is not None}

    @classmethod
    def from_api_response(cls, response: dict):
        """Parse the API response."""
        role = response['choices'][0]['message']['role']
        content = response['choices'][0]['message'].get('content', '')
        name = response['choices'][0]['message'].get('name', None)
        function_call = response['choices'][0]['message'].get('function_call', None)
        if function_call is not None:
            function_call = function_call.to_dict()

        content = '' if content is None else content
        return cls(role, content, name, function_call)

    @classmethod
    def from_function_call(cls, function_name: str, function_output: Any):
        """Prepare a message from the output of a function."""
        return cls('function', str(function_output), function_name)


def start_chat(model: str) -> Callable:
    """Make an LLM interface function that you can use with Messages."""

    def chat_func(messages: List[Message], *args, **kwargs) -> Message:
        input_messages = [message.as_dict() for message in messages]
        try:
            response = oai.ChatCompletion.create(
                messages=input_messages,
                model=model,
                *args,
                **kwargs
            )
            return Message.from_api_response(response)
        except oai.APIError:
            return Message('system', 'There was an API error.  Please try again.')

    return chat_func
