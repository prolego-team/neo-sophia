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
    def __init__(self, db_dir: str, api_key: str):
        """ """

        self.db_dir = db_dir

        oaiapi.set_api_key(api_key)

        chroma.configure_db(self.db_dir)
        self.client = chroma.get_inmemory_client()

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=api_key, model_name=EMBEDDING_MODEL)

        self.pdf_collection = self.client.get_or_create_collection(
            name='pdf_collection',
            embedding_function=openai_ef
        )

        self.page_collection = self.client.get_or_create_collection(
            name='page_collection',
            embedding_function=openai_ef
        )

        self.section_collection = self.client.get_or_create_collection(
            name='section_collection',
            embedding_function=openai_ef
        )

    def get_context_from_query(self, query: str, n_results=4):
        """
        Given a query, this function searches across all sections in the
        database for the closest match, then returns the context around that
        match in the form of a prompt for a LLM.
        """

        # Get the top n results matching the query
        res = self.section_collection.query(
            query_texts=[query], n_results=n_results)

        # Extract the metadata from each result so we can trace back if needed
        # This also filters out sections on the same page
        data = []
        page_nums = []
        for metadata in res['metadatas'][0]:
            if metadata['page_num'] not in page_nums:
                page_nums.append(metadata['page_num'])

                page_res = self.page_collection.get(
                    metadata['filename'] + '~' + str(metadata['page_num']))

                data.append(
                    (
                        metadata['filename'],
                        metadata['page_num'],
                        metadata['section_id'],
                        page_res['documents'][0]
                    )
                )

        prompts = []
        for context in data:
            prompt = f"| Filename: {context[0]} |\n"
            prompt += f"| Page Number: {context[1]} |\n"
            prompt += f"Text: {context[3]}"
            prompts.append(prompt)

        return prompts

    def add_pdf(self, filename):
        """
        Adds a PDF if it doesn't exist in the database. If it does exist,
        update it

        General rule of thumb is 1 token per 4 characters of text
        ref: https://platform.openai.com/tokenizer

        1. Extract flat text from PDF into a list of pages
        2. Create an embedding for each page
        3. Split pages into sections and create an embedding for each section
        4. Average section embeddings to obtain full PDF embedding
        """

        if not os.path.isfile(filename):
            raise ValueError('File not found')

        print(f'Adding {filename} to database...')

        # List of text for each page in the PDF
        page_texts = pdf_utils.extract_text_from_pdf(filename)

        # Get page ids and metadata
        page_ids = [
            filename + '~' + str(page_num) for page_num in range(
                0, len(page_texts)
            )
        ]
        page_metadatas = []
        for page_num in range(0, len(page_texts)):
            page_metadatas.append(
                {
                    'filename': filename,
                    'page_num': page_num
                }
            )

        # Add whole pages to collection
        self.page_collection.upsert(
            ids=page_ids, documents=page_texts, metadatas=page_metadatas)

        # Split each page into 4 sections
        page_sections = []
        for page_text in page_texts:
            page_sections.append(
                text_utils.split_text_into_sections(page_text, 4))

        # Get section ids and metadata
        section_ids = []
        section_texts = []
        section_metadatas = []
        for page_num, sections in enumerate(page_sections):
            for section_id, section_text in enumerate(sections):
                section_ids.append(
                    filename + '~' + str(page_num) + '~' + str(section_id))
                section_texts.append(section_text)
                section_metadatas.append(
                    {
                        'filename': filename,
                        'page_num': page_num,
                        'section_id': section_id
                    }
                )

        # Add sections to collection
        self.section_collection.upsert(
            ids=section_ids, documents=section_texts, metadatas=section_metadatas)

        # Compute the average embedding across all sections for full pdf
        section_embeddings = self.section_collection.get(
            include=['embeddings'])['embeddings']
        section_embeddings = torch.tensor(section_embeddings)
        pdf_embedding = torch.mean(section_embeddings, axis=0)

        # Add full pdf to collection
        self.pdf_collection.upsert(
            ids=[filename],
            documents='\n'.join(page_texts),
            embeddings=[pdf_embedding.tolist()],
            metadatas={
                'filename': filename
            }
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


