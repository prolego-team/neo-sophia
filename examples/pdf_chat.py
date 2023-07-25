"""
"""
import os
import time
import pickle

from collections import Counter

import click
import torch
import numpy as np

from tqdm import tqdm

from examples import project
from neosophia.llmtools import openaiapi as oaiapi, pdf_utils

MAX_TOKENS = {
    'gpt-4': 8192,
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-16k': 16385
}


def split_page_into_sections(page_text, n):
    page_length = len(page_text)
    section_length = page_length // n
    sections = [
        page_text[i:i+section_length] for i in range(
            0, page_length, section_length)
    ]
    return sections


def compute_pdf_embeddings(text_data, n_sections):
    pdf_embeddings = {}
    print('Computing embeddings...')
    for page_num, page_text in tqdm(text_data.items()):
        sections = split_page_into_sections(page_text, n_sections)
        for sidx, text in enumerate(sections):
            emb = oaiapi.extract_embeddings(oaiapi.embeddings(text))[0]
            pdf_embeddings[f'{page_num}-{sidx}'] = emb
    return pdf_embeddings


@click.command()
@click.option(
    '--llm', default='gpt-4',
    type=click.Choice(['gpt-4', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo']))
@click.option(
    '--filepath', '-f', help='',
    default=f'{project.DATASETS_DIR_PATH}/Title-12/Title-12-Volume-1.pdf')
@click.option('--n_sections', '-n', type=int)
def main(llm, filepath, n_sections):

    n_tokens = MAX_TOKENS[llm]

    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    oaiapi.set_api_key(api_key)

    filepath = os.path.normpath(filepath)
    _text_data = pdf_utils.extract_text_from_pdf(filepath)

    text_data = {}
    for page_num, text in _text_data.items():
        if page_num < 20 or page_num > 30:
            continue
        text_data[page_num] = text

    question = 'When does an obligation qualify as a Type I security?'
    question_emb = oaiapi.extract_embeddings(oaiapi.embeddings(question))[0]

    section_embeddings = compute_pdf_embeddings(text_data, n_sections)

    scores = []
    for idx, emb in section_embeddings.items():
        score = torch.sum(emb * question_emb)
        scores.append(score.item())

    n = 5
    n_scores = list(reversed(list(np.sort(scores)[-n:])))
    idxs = list(reversed(list(np.argsort(scores)[-n:])))

    page_nums = []
    keys = list(section_embeddings.keys())
    for idx in idxs:
        page_nums.append(keys[idx])

    for page_num in page_nums:
        text = text_data[int(page_num.split('-')[0])]
        prompt = 'Determine if the following context has enough information to answer the question. If it doesn\'t, say "There is not enough context to provide an answer". If it does, provide the answer. THINK STEP BY STEP. ALWAYS PROVIDE QUOTES AND PAGE CITATIONS.\n\n'
        prompt += f'Question: {question}\n'
        prompt += f'Context: \n{text}'

        out = oaiapi.chat_completion(
            prompt=prompt,
            model=llm)

        if 'There is no' in out:
            print(f'Not enough information')
        else:
            print('prompt:\n')
            print(prompt, '\n')
            print('PAGE_NUM:', page_num)
            print(out)
            print('\n-----------------------------------------------------------\n')
            input()


if __name__ == '__main__':
    main()

