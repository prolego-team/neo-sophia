"""
"""
import os

import chromadb

from chromadb.utils import embedding_functions

import neosophia.db.chroma as chroma

from neosophia.llmtools import openaiapi as oaiapi, pdf_utils, text_utils

EMBEDDING_MODEL = 'text-embedding-ada-002'


class PDFDB:
    def __init__(self, db_dir: str):
        self.db_dir = db_dir

        chroma.configure_db(self.db_dir)
        self.chroma_client = chroma.get_inmemory_client()

        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=os.getenv('OPENAI_API_KEY'),
                        model_name=EMBEDDING_MODEL
        )

        self.page_collection = self.chroma_client.get_or_create_collection(
            name='page_collection',
            embedding_function=self.openai_ef
        )

        self.section_collection = self.chroma_client.get_or_create_collection(
            name='section_collection',
            embedding_function=self.openai_ef
        )

    def add_pdf(self, filename):
        """ """

        if not os.path.isfile(filename):
            raise ValueError('File not found')

        # List of text for each page in the PDF
        pdf_text = pdf_utils.extract_text_from_pdf(filename)

        page_data_dict = {}
        section_data_dict = {}
        for page_num, page_text in enumerate(pdf_text):

            # Add the embedding for the whole page
            page_data_dict[f'{page_num}'] = {
                'text': page_text,
                'embedding': oaiapi.extract_embeddings(
                    oaiapi.embeddings(page_text))[0]
            }

            # Split the page into 4 parts and add all those embeddings
            page_sections = text_utils.split_text_into_sections(page_text, 4)
            for sidx, text in enumerate(page_sections):
                section_data_dict[f'{page_num}-{sidx}'] = {
                    'text': text,
                    'embedding': oaiapi.extract_embeddings(
                        oaiapi.embeddings(text))[0]
                }

        # Add whole pages to collection
        for page_num, data in page_data_dict.items():

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

    def get_section(
            self, filename, page_num, section_id, include_embeddings=False):
        include = ['documents', 'metadatas']
        if include_embeddings:
            include.append('embeddings')
        return self.section_collection.get(
            filename + '-' + str(page_num) + '-' + str(section_id), include)

    def get_all_pdfs(self):
        return self.db.find_all(self.collection_name)

    def delete_pdf(self, filename):
        query = {'filename': filename}
        self.db.delete(self.collection_name, query)



