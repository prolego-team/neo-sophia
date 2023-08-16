""" Example script to interact with the Agent class """
from examples import project

from neosophia.db import sqlite_utils as sql_utils
from neosophia.llmtools import openaiapi as oaiapi
from neosophia.agents.base import Agent
from neosophia.agents.system_prompts import UNLQ_GPT_BASE_PROMPT


def main():
    """ main """

    oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)

    modules = []
    resources = [
        'data/cats_and_dogs.db',
        'data/synthbank.db',
        'data/transactions.db'
    ]

    system_prompt = UNLQ_GPT_BASE_PROMPT
    agent = Agent('MyAgent', system_prompt, modules, resources)
    agent.chat()


if __name__ == '__main__':
    main()
