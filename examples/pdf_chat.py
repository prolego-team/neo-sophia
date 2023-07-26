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

CHROMADB_PERSIST_DIR = '.chroma_cache'
MAX_TOKENS = {
    'gpt-4': 8192,
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-16k': 16385
}



def compute_pdf_embeddings(text_data, n_sections):
    pdf_embeddings = {}
    print('Computing embeddings...')
    for page_num, page_text in tqdm(text_data.items()):
        sections = text_utils.split_text_into_sections(page_text, n_sections)
        for sidx, text in enumerate(sections):
            emb = oaiapi.extract_embeddings(oaiapi.embeddings(text))[0]
            pdf_embeddings[f'{page_num}-{sidx}'] = {
                'text': text,
                'embedding': emb
            }
    return pdf_embeddings


@click.command()
@click.option(
    '--llm', default='gpt-4',
    type=click.Choice(['gpt-4', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo']))
@click.option(
    '--data_dir', '-f', help='Path to a directory containing PDFs',
    default=f'{project.DATASETS_DIR_PATH}/Title-12')
@click.option('--n_sections', '-n', type=int, default=4)
def main(llm, data_dir, n_sections):
    """ """
    data_dir = os.path.normpath(data_dir)
    filepaths = pdf_utils.find_pdfs_in_directory(data_dir)
    n_tokens = MAX_TOKENS[llm]
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    oaiapi.set_api_key(api_key)

    question = 'When does an obligation qualify as a Type I security?'
    question_emb = oaiapi.extract_embeddings(oaiapi.embeddings(question))[0]

    # Setup database
    chroma.configure_db(str(CHROMADB_PERSIST_DIR))
    chroma_client = chroma.get_inmemory_client()

    # Extract text from the PDF
    text_data = pdf_utils.extract_text_from_pdf(filepaths[0])

    # Split up into sections
    section_text_data = text_utils.split_text_into_sections(
        text_data, n_sections)

    section_embeddings = compute_pdf_embeddings(text_data, n_sections)

    collection = client.create_collection(name='title-12-emb')

    for ps, data in section_embeddings.items():
        page_num, section_id = ps.split('-')
        text = data['text']
        embedding = data['embedding'].tolist()

        collection1.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[{'page': page_num}],
            ids=[page_num + '-' + section_id]
        )

    res = collection1.query(
        query_embeddings=[question_emb.tolist()], n_results=4)
    print('res:', res)

    exit()

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

