"""
"""
import os
import sqlite3

import click

from examples import project

from neosophia.db import sqlite_utils as sql_utils
from neosophia.llmtools import dispatch as dp, openaiapi as oaiapi
from neosophia.agents.base import Agent
from neosophia.agents.system_prompts import (FUNCTION_GPT_PROMPT,
                                             UNLQ_GPT_BASE_PROMPT)

opj = os.path.join


def setup():
    db_file = opj(project.DATASETS_DIR_PATH, 'synthbank.db')
    conn = sqlite3.connect(db_file)
    conn = create_cats_and_dogs_db('data/cats_and_dogs.csv')
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    return api_key, conn


def create_cats_and_dogs_db(csv_file: str):
    db_file = opj(project.DATASETS_DIR_PATH, 'cats_and_dogs.db')
    conn = sqlite3.connect(db_file)
    sql_utils.create_database_from_csv(conn, csv_file, 'pets')
    return conn


def main():
    """
    """

    api_key, conn = setup()

    modules = [sql_utils]
    resources = [
        'data/synthbank.db',
        'data/transactions.db',
        'data/cats_and_dogs.db'
    ]

    prompt = UNLQ_GPT_BASE_PROMPT

    agent = Agent('MyAgent', prompt, modules, resources)
    agent.chat()

    exit()

    print(prompt)
    print('\n-----------\n')
    out = agent.execute(prompt)
    print('OUT')
    print(out)
    exit()
    agent.chat()

    exit()


if __name__ == '__main__':
    main()
