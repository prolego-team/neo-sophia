"""
"""
import os

from typing import List
from dataclasses import dataclass

import torch
import chromadb

from tqdm import tqdm
from chromadb.utils import embedding_functions

import neosophia.db.chroma as chroma

from neosophia.llmtools import openaiapi as oaiapi, pdf_utils, text_utils

EMBEDDING_MODEL = 'text-embedding-ada-002'

MAX_INPUT_TOKENS = {
    'text-embedding-ada-002': 8191
}


class PDFDB:
    def __init__(self, db_dir: str):
        self.db_dir = db_dir

        chroma.configure_db(self.db_dir)
        self.chroma_client = chroma.get_inmemory_client()

        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=os.getenv('OPENAI_API_KEY'),
                        model_name=EMBEDDING_MODEL
        )

        self.pdf_collection = self.chroma_client.get_or_create_collection(
            name='pdf_collection',
            embedding_function=self.openai_ef
        )

        self.page_collection = self.chroma_client.get_or_create_collection(
            name='page_collection',
            embedding_function=self.openai_ef
        )

        self.section_collection = self.chroma_client.get_or_create_collection(
            name='section_collection',
            embedding_function=self.openai_ef
        )

    def search_by_embedding(self, query: str, n_results: int):
        query_embedding = oaiapi.extract_embeddings(oaiapi.embeddings(query))[0]
        return self.section_collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=n_results)

    def get_context(self, query: str):
        """ """

        res = self.search_by_embedding(query, n_results=8)

        pages = []
        for rid in res['ids'][0]:
            filename, page_num, _ = rid.split('~')
            if page_num not in pages:
                pages.append(page_num)

        data = []
        for page_num in pages:
            page_res = self.page_collection.get(
                filename + '~' + str(page_num))
            data.append(
                {
                    'text': page_res['documents'][0],
                    'filename': filename,
                    'page_num': page_num
                }
            )

        return data

    def add_pdf(self, filename):
        """
        Adds a PDF if it doesn't exist in the database. If it does exist,
        update it

        General rule of thumb is 1 token per 4 characters of text
        ref: https://platform.openai.com/tokenizer

        1. Extract flat text from PDF into a list of pages
        2. Create an embedding for each page
        section
        3. Split pages into sections and create an embedding for each section
        4. Split entire PDF into sections of length `max_chars`
        6. Create an embedding for each page, then average the embeddings to
        create a full document embedding
        6. Another option would be to recursively split and summarize the
        document into chunks of size `max_chars`, then embed the final summary
        """

        if not os.path.isfile(filename):
            raise ValueError('File not found')

        print(f'Adding {filename} to database...')

        # List of text for each page in the PDF
        page_texts = pdf_utils.extract_text_from_pdf(filename)

        '''
        # Maximum number of characters we can send to the embedding model
        max_chars = MAX_INPUT_TOKENS['text-embedding-ada-002'] * 4
        print('max_chars:', max_chars)

        all_text = ' '.join([pt for pt in page_texts])
        print('all_text:', len(all_text))

        num_large_sections = int(len(all_text) / max_chars) + 1
        print('num_large_sections:', num_large_sections)

        large_sections = text_utils.split_text_into_sections(
            all_text, num_large_sections)

        prompt = 'Give me the TLDR:\n\n'
        summarized_sections = []
        for text in tqdm(large_sections):
            summarized_sections.append(
                oaiapi.chat_completion(
                    prompt=prompt + text,
                    model='gpt-4'))
        summarized_sections = '\n'.join(summarized_sections)
        '''

        # Get page and section embeddings
        page_data, section_data_dict = text_utils.create_page_embeddings(
            page_texts)

        # Add whole pages to collection
        for page_num, data in enumerate(page_data):
            self.page_collection.upsert(
                embeddings=[data['embedding'].tolist()],
                documents=[data['text']],
                metadatas=[
                    {
                        'filename': filename,
                        'page_num': page_num
                    }
                ],
                ids=[filename + '~' + str(page_num)]
            )

        # Compute the average embedding across all sections
        pdf_embedding = torch.mean(
            torch.cat(
                [
                    torch.unsqueeze(
                        x['embedding'], 0
                    ) for x in section_data_dict.values()
                ]),
            axis=0
        )

        # Add full pdf to collection
        self.pdf_collection.upsert(
            embeddings=[pdf_embedding.tolist()],
            documents=['\n'.join(page_texts)],
            metadatas=[{'filename': filename}],
            ids=[filename]
        )

        # Add sections to collection
        for page_id, data in section_data_dict.items():
            page_num, section_id = page_id.split('-')
            self.section_collection.upsert(
                embeddings=[data['embedding'].tolist()],
                documents=[data['text']],
                metadatas=[
                    {
                        'filename': filename,
                        'page_num': page_num,
                        'section_id': section_id
                    }
                ],
                ids=[filename + '~' + str(page_num) + '~' + str(section_id)]
            )

    def get_pdf(self, filename):
        return self.pdf_collection.get(filename)

    def get_page(self, filename, page_num, include_embeddings=False):
        include = ['documents', 'metadatas']
        if include_embeddings:
            include.append('embeddings')
        return self.page_collection.get(
            filename + '~' + str(page_num), include=include)

    def get_section(
            self, filename, page_num, section_id, include_embeddings=False):
        include = ['documents', 'metadatas']
        if include_embeddings:
            include.append('embeddings')
        return self.section_collection.get(
            filename + '~' + str(page_num) + '~' + str(section_id),
            include=include)


