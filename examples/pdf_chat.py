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

MAX_TOKENS = {
    'gpt-4': 8192,
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-16k': 16385
}


def format_page_res(context):
    """ """
    prompt = f"Filename: {context['filename']}\n"
    prompt += f"Page Number: {context['page_num']}\n"
    prompt += f"Text: {context['text']}"
    return prompt


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

    # Create database
    pdfdb = PDFDB('pdf_db')

    #filename1 = 'data/Title-12-Small/out.pdf'
    filename1 = 'data/Title-12/Title-12-Volume-1.pdf'
    filename2 = 'data/2306.05284.pdf'
    #pdfdb.add_pdf(filename1)
    #pdfdb.add_pdf(filename2)

    question = 'What is the difference between a Type I and Type II security?'
    result = pdfdb.get_context(question)

    '''
    out = pdfdb.section_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=4,
    )

    page = pdfdb.get_page(filename=filename1, page_num=1)
    out = pdfdb.get_section(
        filename=filename1, page_num=0, section_id=0)
    res = pdfdb.section_collection.query(
        query_embeddings=[question_emb.tolist()], n_results=4)
    '''

    for res in result:

        prompt = 'Determine if the following context has enough information to answer the question. If it doesn\'t, say "There is not enough context to provide an answer". If it does, provide the answer. THINK STEP BY STEP. ALWAYS PROVIDE QUOTES AND PAGE CITATIONS.\n\n'
        prompt += f'Question: {question}\n'

        context = format_page_res(res)
        prompt += f'Context: \n{context}'

        out = oaiapi.chat_completion(
            prompt=prompt,
            model=llm)

        if 'There is no' in out:
            print(f'Not enough information on page', str(res['page_num']))
        else:
            print('Answer found on page', res['page_num'])
            print(out)
            print('\n-----------------------------------------------------------\n')


if __name__ == '__main__':
    main()

