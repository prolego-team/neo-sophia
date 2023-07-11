"""Example of how to use llamaindex for semantic search.

This example assumes that initially there is a projects.DATASETS_DIR_PATH/embeddings.pkl file
that has a list of dictionaries with each dictionary containing "text",
"rule_name" and "section_label" fields.

The first time you run this script, a vector store will be creaed with
embeddings.  This store will be saved to "cache/msrb_index_store".
Subsequent runs will load the vector store from this location.

Each time you run this script you enter a loop where you can ask as
many questions of the data as you'd like.  Each time you ask a question
you will be given a response that tells you:

1. The rule names and section labels for the most relevant rules,
2. A brief preview of the text from those sections, and
3. An LLM-generated response to your query given the texts that it found.

You can tweak three parameters at the bottom of this script (after all of
the function definitions):

- model_name: which OpenAI model to use.
- top_k: how many rules to return.
- similarity_cutoff: threshold for relevance (between 0 and 1).
"""


import os
import pickle
from pathlib import Path

# from llama_index import SimpleDirectoryReader
# from llama_index.node_parser import SimpleNodeParser
from llama_index import (
    VectorStoreIndex,
    StorageContext,
    LLMPredictor,
    ServiceContext,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.schema import TextNode

from langchain import OpenAI

from examples import project


TEXT_DATA_FILE = Path(os.path.join(project.DATASETS_DIR_PATH, 'embeddings.pkl'))
INDEX_DATA_DIR = Path('cache/msrb_index_store')


def get_vector_store(service_context: ServiceContext) -> VectorStoreIndex:
    """Load a vector index from disk or, if it doesn't exist, create one from raw text data."""

    # === Load the data ===========================================================
    # Simple example of reading text files from a local directory
    # reader = SimpleDirectoryReader('./data')
    # documents = reader.load_data()  # returns a list of Documents
    # parser = SimpleNodeParser()
    # nodes = parser.get_nodes_from_documents(documents) # returns a list of nodes

    if INDEX_DATA_DIR.exists():
        print('Loading vector store from local directory.')
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DATA_DIR)
        # load index
        index = load_index_from_storage(storage_context)
    else:
        print('No local index found.')
        print('Loading data.')
        with open('embeddings.pkl', 'rb') as f:
            data = pickle.load(f)

        print('Building nodes.')
        nodes = []
        for example in data:
            node = TextNode(text=example['text'])
            node.metadata['rule_name'] = example['rule_name']
            node.metadata['section_label'] = example['section_label']
            nodes.append(node)
        print(f'Created {len(nodes)} nodes.')

        print('Creating vector store.')
        index = VectorStoreIndex(nodes, service_context=service_context)
        # index = VectorStoreIndex.from_documents(documents)
        print('Saving vector store.')
        index.storage_context.persist(INDEX_DATA_DIR)

    return index


def get_llm_backend(model_name: str) -> ServiceContext:
    """Get an LLM to provide embedding and text generation service."""

    # === Define the LLM backend ==================================================
    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=model_name))

    # configure service context
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    return service_context


def get_query_engine(index: VectorStoreIndex, response_mode: str, top_k: int, similarity_cutoff: float) -> RetrieverQueryEngine:
    """Build a query enginge by combining a retriever and response synthesizer."""
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    # query_engine = RetrieverQueryEngine.from_args(
    #     retriever=retriever,
    #     response_synthesizer=response_synthesizer,
    #     response_mode=response_mode
    # )
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        ]

    )

    return query_engine


if __name__=='__main__':

    model_name = "text-davinci-003"
    top_k = 3
    similarity_cutoff = 0.7

    service_context = get_llm_backend(model_name)
    index = get_vector_store(service_context)

    response_mode = 'refine'  # response_mode = 'no_text' for no text generation
    query_engine = get_query_engine(index, response_mode, top_k, similarity_cutoff)

    # query
    while (query := input('Ask me a question about the MSRB rule book ("quit" to quit): ')) != 'quit':
        print(f'You asked: {query}')
        response = query_engine.query(query)
        print('Source nodes:')
        print(f'There are {len(response.source_nodes)} source nodes from the following rules:')
        for source_node in response.source_nodes:
            print(source_node.node.metadata['rule_name'], source_node.node.metadata['section_label'])
        print(response.get_formatted_sources())

        print('Response:')
        print(response)
        print()
        print('='*40)
