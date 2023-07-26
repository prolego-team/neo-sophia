"""
"""
import os
import time
import pickle

from collections import Counter

import click
import torch
import chromadb
import numpy as np

from tqdm import tqdm

import neosophia.db.chroma as chroma

from examples import project

from neosophia.llmtools import openaiapi as oaiapi, pdf_utils, text_utils
from neosophia.db.pdfdb import PDFDB

CHROMADB_PERSIST_DIR = '.chroma_cache'
MAX_TOKENS = {
    'gpt-4': 8192,
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-16k': 16385
}


@click.command()
@click.option(
    '--llm', default='gpt-4',
    type=click.Choice(['gpt-4', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo']))
@click.option(
    '--data_dir', '-f', help='Path to a directory containing PDFs',
    default=f'{project.DATASETS_DIR_PATH}/Title-12')
def main(llm, data_dir):
    """ """

    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    oaiapi.set_api_key(api_key)

    question = 'When does an obligation qualify as a Type I security?'
    question_emb = oaiapi.extract_embeddings(oaiapi.embeddings(question))[0]

    filename = 'data/Title-12-Small/out.pdf'
    pdfdb = PDFDB('pdf_db')
    pdfdb.add_pdf(filename)

    out = pdfdb.get_section(
        filename=filename, page_num=0, section_id=0)

    print('OUT:', out)
    exit()

    res = pdfdb.section_collection.query(
        query_embeddings=[question_emb.tolist()], n_results=4)

    pages = []
    for rid in res['ids'][0]:
        _, page_num, _ = rid.split('~')
        if page_num not in pages:
            pages.append(page_num)

    for page_num in pages:

        page_res = pdfdb.page_collection.get(
            filename + '~' + str(page_num))

        text = page_res['documents'][0]
        prompt = 'Determine if the following context has enough information to answer the question. If it doesn\'t, say "There is not enough context to provide an answer". If it does, provide the answer. THINK STEP BY STEP. ALWAYS PROVIDE QUOTES AND PAGE CITATIONS.\n\n'
        prompt += f'Question: {question}\n'
        prompt += f'Context: \n{text}'

        out = oaiapi.chat_completion(
            prompt=prompt,
            model=llm)

        if 'There is no' in out:
            print(f'Not enough information on page', str(page_num))
        else:
            print('Answer found on page', page_num)
            print(out)
            print('\n-----------------------------------------------------------\n')
            break


if __name__ == '__main__':
    main()

