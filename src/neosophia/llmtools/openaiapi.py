"""
Wrappers for OpenAI API.
"""

from typing import Any, List, Callable, Optional
import os
import sys
from functools import partial
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
    """Data structure for working with the messages that are passed back
    and forth to the OpenAI chat APIs."""
    role: str
    content: str

    def as_dict(self) -> dict:
        """Because it's much, much faster than built in asdict."""
        return vars(self)

    @classmethod
    def from_api_response(cls, response: dict):
        """Parse the API response."""
        role = response['choices'][0]['message']['role']
        content = response['choices'][0]['message']['content']
        return cls(role, content)


def start_chat(model: str) -> Callable:
    """Make an LLM interface function that you can use with Messages."""

    def chat_func(messages: List[Message], *args, **kwargs) -> Message:
        input = [message.as_dict() for message in messages]
        response = oai.ChatCompletion.create(messages=input, model=model, *args, **kwargs)
        return Message.from_api_response(response)
    
    return chat_func

