import sys
import os
import logging
import pickle
from pathlib import Path

from chromadb.utils import embedding_functions

import neosophia.db.chroma as chroma

from examples import project


# === Config settings ===============================================
TEXT_DATA_FILE = Path(os.path.join(project.DATASETS_DIR_PATH, 'embeddings.pkl'))
CHROMADB_PERSIST_DIR = Path('.chroma_cache')
EMBEDDING_MODEL = 'text-embedding-ada-002'


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
chroma.configure_db(str(CHROMADB_PERSIST_DIR))
chroma_client = chroma.get_inmemory_client()

log.info(f'This database has the following collections:')
for collection in chroma_client.list_collections():
    log.info(f'  Name = {collection.name}, count = {collection.count()}')

log.info(f'Getting msrb_rules and updating/inserting records.')
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name=EMBEDDING_MODEL
)
collection = chroma_client.get_or_create_collection(
    name='msrb_rules',
    embedding_function=openai_ef
)
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
