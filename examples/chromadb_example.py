import sys
import os
import logging
import pickle
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


# === Config settings ===============================================
TEXT_DATA_FILE = Path('data/embeddings.pkl')
CHROMADB_PERSIST_DIR = Path('.chroma_cache')


# === Basic setup ===================================================
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logging.getLogger('chroma_ex').setLevel(logging.DEBUG)
log = logging.getLogger('chroma_ex')


# === Load data =====================================================
log.info('Loading data.')
with open(TEXT_DATA_FILE, 'rb') as f:
    data = pickle.load(f)
log.info(f'Loaded {len(data)} sections of text.')

texts = [example['text'] for example in data]
embeddings = [example['emb'].tolist() for example in data]
metadata = [{
    'rule_name': example['rule_name'],
    'section_label': ' '.join(example['section_label']) if example['section_label'] != (None,) else ''
} for example in data]


# === Database stuff ================================================
log.info('Creating ChromaDB client with OpenAI embeddings.')
chroma_client = chromadb.Client(Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=str(CHROMADB_PERSIST_DIR)
))
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name='text-embedding-ada-002'
)
log.info(f'This database has the following collections: {chroma_client.list_collections()}')

log.info(f'Getting msrb_rules and updating/inserting records.')
collection = chroma_client.get_or_create_collection(name='msrb_rules', embedding_function=openai_ef)
collection.upsert(
    embeddings=embeddings,
    documents=texts,
    metadatas=metadata,
    ids=[str(i) for i in range(len(texts))]
)

log.info('Running query.')
results = collection.query(
    query_texts=["How should a broker confirm a transaction with a customer?"],
    n_results=2
)
print(results)

log.info('Persisting databse')
chroma_client.persist()