"""FIA Regulation Search for Formula One

## Key concepts:

- Regulations have been parsed as DocTrees
- DocTrees are simply hierarchical data structures storing text with some metadata
- Some regulations had definition lists that were also extracted
- There are three kinds of searches

    - Quick search: simple semantic search over regulations and definitions
    - Search and summarize: "Quick search" + LLM summary of results
    - Agentic search: Quick search + ability for LLM to follow up/clarify with
      additional questions against the quick search function.

- Regulation search is performed by minimizing the cosine distance between the embedded
  query and the embedded regulation texts.
- Regulation search can be augmented with a cross encoder reranker.
- Definition searches use BM25, since keywords are more important.
- Each query is also scanned for Capitalized Phrases and the definitions are searched
  for a beginning-of-string exact match against the phrases.
"""
import os
from pathlib import Path
import re
import json
import logging
import sys

from sentence_transformers import CrossEncoder
import gradio as gr

from neosophia.llmtools import openaiapi as openai

import neosophia.search.embeddings as emb
import neosophia.search.utils.doctree as doctree

from neosophia.search.utils import tree
from neosophia.search.semantic_search import cosine_search, rerank
from neosophia.search.keyword_search import keyword_search, build_index
from neosophia.search.utils.data_utils import get_dict_hash, SearchResult, reciprocal_rank_fusion
from neosophia.agents.react_chat import make_react_agent
from neosophia.text_utils import get_capitalized_phrases

from examples import project
from examples.fia.utils import (
    load_memos,
    load_regs,
    load_defs,
    get_embeddings
)

# Suppress a runtime warning re: tokenizer parallelism and multiple threads.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Basic logger setup ===================================================
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logging.getLogger('search').setLevel(logging.DEBUG)
log = logging.getLogger('search')
# ===================================================================

DOC_DIR = Path('data/fia/docs')
DATA_DIR = Path('data/fia/data')

RERANK = True
PRE_EXPAND = False
POST_EXPAND = True

REGS = {
    '2023 FIA Formula One Sporting Regulations': 'fia_2023_formula_1_sporting_regulations_-_issue_6_-_2023-08-31.yaml',
    '2023 FIA International Sporting Code': '2023_international_sporting_code_fr-en_clean_9.01.2023.yaml',
    '2023 FIA International Sporting Code, Appendix L, Chapter II': 'appendix_l_iii_2023_publie_le_20_juin_2023.yaml',
    '2023 FIA International Sporting Code, Appendix L, Chapter IV': 'appendix_l_iv_2023_publie_le_20_juin_2023.yaml',
    '2023 FIA Formula One Financial Regulations': 'fia_formula_1_financial_regulations_-_issue_16_-_2023-08-31.yaml',
    '2023 FIA Formula One Technical Regulations': 'fia_2023_formula_1_technical_regulations_-_issue_7_-_2023-08-31.yaml'
}
MEMO_DIRS = {
    '2023 FIA Formula One Sporting Regulations': 'sporting',
    '2023 FIA Formula One Financial Regulations': 'financial'
}

MAX_LLM_CALLS_PER_INTERACTION = 5

FUNCTION_DESCRIPTIONS = [
    {
        'name': 'lookup_definition',
        'description': 'Lookup a word or phrase in the glossary to get its definition.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'A word or phrase for which you want the definition.'
                },
            },
            'required': ['query']
        }
    },
    {
        'name': 'regulation_search',
        'description': 'Search regulations using semantic search.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Search query in the form of a question.'
                },
            },
            'required': ['query']
        }
    }
]

REG_DIVIDER = '\n\n---\n\n'
DEF_DIVIDER = '\n\n'


def result_to_string(result: SearchResult, doc_trees: dict[str, doctree.DocTree]) -> str:
    """Format a search result as a string with context."""
    file = result.file
    section_ind = result.tree_index

    section = tree.get_from_tree(doc_trees[file], section_ind)
    text = result.text

    top_ind = section_ind
    section_headings = [section.title,]
    while len(next_up:=tree.move_up(top_ind))>0:
        top_ind = next_up
        section_headings.append(tree.get_from_tree(doc_trees[file], top_ind).title)

    section_headings = section_headings[::-1]
    if len(section_headings)<2:
        section_headings.append('None')

    # get additional context
    text = f'**{text}**'
    super_text = doctree.get_supersection(doc_trees[file], section_ind)
    if super_text is not None:
        text = super_text + '\n\n' + text

    sub_text = doctree.get_subsection(doc_trees[file], section_ind)
    if sub_text is not None:
        text = text + '\n\n' + sub_text

    # uri = DOC_DIR / f'{result.metadata["file"]}'
    # uri = uri.absolute().as_uri() + f'#page={result.metadata["page"]}'

    summary_str = (
        f'{file.title()} Regulation: {", ".join(section_headings)}\n\n'
        f'{text}\n'
    )

    return summary_str


def results_to_string(results: list[SearchResult], doc_trees: dict[str, doctree.DocTree]) -> str:
    """Convert a a list of search results to a string."""
    output = [result_to_string(result, doc_trees) for result in results]
    return REG_DIVIDER.join(output)


def build_context(regulations: str, definitions: str) -> str:
    """Combine regulation and definitions texts for LLM context."""
    context = (
        'Here are some potentially useful definitions:\n\n'
        f'{definitions}'
        '\n\n---\n\n'
        'Here is potentially useful context from the regulations:\n\n'
        f'{regulations}'
    )
    return context


def setup():
    """Setup everything needed for the demo."""

    # Basic configuration parameters
    model_name = 'all-mpnet-base-v2'
    cross_encoder_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    llm_model_name = 'gpt-4'
    config = {
        'pre_expand': PRE_EXPAND,
        'post_expand': POST_EXPAND,
        'similarity_model_name': model_name,
        'cross_encoder_name': cross_encoder_name
    }
    top_k = 10

    run_id = get_dict_hash(config)
    run_dir = DATA_DIR / Path(str(run_id))
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(config, f)

    api_key = openai.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    openai.set_api_key(api_key)


    # Get models
    model = emb.get_model(model_name)
    if RERANK:
        rerank_model = CrossEncoder(cross_encoder_name)

    llm_model = openai.start_chat(llm_model_name)
    system_message = (
        'You are an assitant to a Formula 1 team.  Your job is to answer team questions '
        'to the best of your ability.  You will be given several excerpts from regulations '
        'to help you in answering.  Some of the provided regulations may be irrelevant, so '
        'ignore these and only use those that appear to be relevant. Do not mention or cite '
        'regulations that are not given to you.  Answer succinctly and make reference to the '
        'relevant regulation sections.\n\n'
        'Think carefully about the question and the provided regulations and definitions. Check that your '
        'response makes sense before answering.'
    )

    # Load data
    log.info('Loading regulations')
    doc_trees = load_regs(REGS, DOC_DIR)

    log.info('Loading definitions')
    definitions_flat, definition_ids = load_defs(REGS, DOC_DIR)

    # Get encodings
    log.info('Getting encodings for regs')
    embeddings, flat_texts, flat_ids = get_embeddings(doc_trees, run_dir, model, PRE_EXPAND)
    log.info(f'Embeddings -- {type(embeddings)} -- {embeddings.shape}')

    definition_bm25 = build_index(definitions_flat)

    # Wrapper functions for search and generation
    def search_definitions(query: str) -> list[SearchResult]:
        """Do a keyword search over definitions."""
        log.info(f'Searching definitions: {query[:20]}...')
        results = keyword_search(definition_bm25, query, definition_ids, definitions_flat, 5)

        return results


    def search_regulations(query: str) -> list[SearchResult]:
        """Search regulation embeddings."""
        log.info(f'Searching regulations: {query[:20]}...')
        query_emb = emb.encode(query, model)

        results = cosine_search(query_emb, embeddings, flat_ids, flat_texts, top_k)
        if RERANK:
            results = rerank(results, doc_trees, query, rerank_model, post_expand=POST_EXPAND)

        # Apply a threshold to results
        if RERANK:
            results = [result for result in results if result.reranked_score>-2]
        else:
            results = [result for result in results if result.similarity_score>0.3]
        log.debug(f'Found {len(results)} regulation results.')

        return results


    def search(query: str) -> str:
        """Do a semantic search over embeddings.  Also return potentially relevant definitiosn."""
        regulation_results = search_regulations(query)
        regulation_results_str = results_to_string(regulation_results, doc_trees)

        # Get definitions that may be semantically relevant to the query
        query_definitions = search_definitions(query)
        query_definitions = [
            f'{defn_hit.text} (from {defn_hit.file})' for defn_hit in query_definitions
        ]
        log.debug(f'Found {len(query_definitions)} query definitions')

        # Look for capitalized phrases from the regulations in the definitions
        # TODO Could speed this up by storing definitions in a dictionary
        phrase_definitions = []
        for res in regulation_results:
            phrases = set(get_capitalized_phrases(res.text))
            if len(phrases)>0:
                for phrase in phrases:
                    if len(phrase)>5:
                        defs = [defn for defn in definitions_flat if re.match(f'"?{phrase}"?', defn)]
                        phrase_definitions += defs
        phrase_definitions = list(set(phrase_definitions))
        log.debug(f'Found {len(phrase_definitions)} phrase definitions')

        # Look for definitions that may be semantically similar to to the regulation results
        regulation_definitions_set = []
        for result in regulation_results:
            regulation_definitions_set.append(search_definitions(result.text))
        regulation_definitions_set = [
            [f'{defn_hit.text} (from {defn_hit.file})' for defn_hit in defn_results]
            for defn_results in regulation_definitions_set
        ]
        regulation_definitions = reciprocal_rank_fusion(regulation_definitions_set)
        log.debug(f'Found {len(regulation_definitions)} regulation definitions')

        definition_results = list(set(regulation_definitions[:5] + phrase_definitions + query_definitions[:2]))
        definition_results_str = DEF_DIVIDER.join(definition_results)

        return regulation_results_str, definition_results_str


    def generate_response(question: str, regulations: str, definitions: str) -> str:
        """Generate a simple response."""
        log.info('Calling LLM')
        context = build_context(regulations, definitions)
        prompt = context + f'\n\nHere is the question: {question}'
        messages = [
            openai.Message('system', system_message),
            openai.Message('user', prompt)
        ]
        return llm_model(messages).content


    def agentic_search(question: str, regulations: str, definitions) -> str:
        """Agent-based search."""

        # Prep agent functions and prompts
        def agent_search_wrapper(query):
            return search(query)[0]

        functions = {
            'lookup_definition': search_definitions,
            'regulation_search': agent_search_wrapper
        }
        system_message_new = (
            system_message +
            '\n\nIf you need additional information, if you need to refine your response, '
            'or if the provided regulations do not appear to answer the question, '
            'you should run additional regulation searches using the `regulation_search` function. '
            'When using this function your queries should rephrase or refine the original '
            'question; don\'t repeat the original question becuase you will get the same results. '
            'You can also look up any word or phrase that you are not sure about using the '
            '`lookup_definition` function.\n\n'
            'When you have the answer write "Final Answer:" followed by the response.'
        )
        context = build_context(regulations, definitions)
        agent = make_react_agent(
            system_message_new,
            llm_model,
            FUNCTION_DESCRIPTIONS,
            functions,
            MAX_LLM_CALLS_PER_INTERACTION,
            extra_context=context
        )

        # Agent loop
        prompt = question + '\n\n' + context
        status = ''
        for message in agent(prompt):
            if message.role=='user':
                status = 'User asked a question to the LLM.  Awaiting LLM response...'
            elif message.role=='function':
                status = 'A query was run against the regulations.  Awaiting LLM response...'
                log.info('Function call: ' + f'\n\n_<name={message.name}, function_call={message.function_call}>_\n')
                response = message.content.replace('Observation: ', '')
                if message.name=='regulation_search':
                    # Add new regulations to exist list
                    regulations_old = regulations.split(REG_DIVIDER)
                    regulations_new = response.split(REG_DIVIDER)
                    regulations = [reg for reg in regulations_new if reg not in regulations_old]
                    regulations += regulations_old
                    regulations = REG_DIVIDER.join(regulations)
                elif message.name=='lookup_definition':
                    definitions = response + DEF_DIVIDER + definitions
            elif message.role=='assistant':
                if 'Final Answer:' in message.content:
                    status = message.content
                else:
                    status = 'The assistant responded.  Awaiting LLM next response...'

            log.info(status)
            yield regulations, definitions, status

    return search, agentic_search, generate_response


def run_demo():

    log.info('Loading memos')
    memo_texts = load_memos(REGS, MEMO_DIRS, DOC_DIR)

    search, agentic_search, generate_response = setup()

    with gr.Blocks() as demo:
        gr.Markdown('# FIA Regulation Matching')
        policy_selector = gr.Dropdown(memo_texts, label='Select a memorandum item:')
        question_text = gr.Textbox('Memo item or question:', label='Search box', interactive=True)
        with gr.Row():
            quick_search_button = gr.Button('Quick Search')
            search_summarize_button = gr.Button('Search and Summarize')
            deep_search_button = gr.Button('Agentic Search')
        llm_response = gr.Textbox('LLM Response', interactive=False, label='LLM Output')
        regulation_texts = gr.Textbox('Output will appear here', interactive=False, label='Retrieved Documents')
        definition_texts = gr.Textbox('Definitions will appear here', interactive=False, label='Definitions')

        policy_selector.change(
            fn=lambda inp: 'Where is this discussed in the FIA regulations: ' + inp.split('::')[-1],
            inputs=policy_selector,
            outputs=question_text
        )

        quick_search_button.click(
            fn=search,
            inputs=question_text,
            outputs=[regulation_texts, definition_texts]
        ).then(
            fn=lambda : 'No LLM response requested.',
            inputs=None,
            outputs=llm_response
        )

        search_summarize_button.click(
            fn=search,
            inputs=question_text,
            outputs=[regulation_texts, definition_texts]
        ).then(
            fn= generate_response,
            inputs=[question_text, regulation_texts, definition_texts],
            outputs=llm_response
        )

        deep_search_button.click(
            fn=search,
            inputs=question_text,
            outputs=[regulation_texts, definition_texts]
        ).then(
            fn=agentic_search,
            inputs=[question_text, regulation_texts, definition_texts],
            outputs=[regulation_texts, definition_texts, llm_response]
        )

    demo.queue()
    demo.launch()


if __name__ == '__main__':
    run_demo()
