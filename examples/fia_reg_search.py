import os
from pathlib import Path
import re
import json
import logging
import sys

import yaml
import torch
from sentence_transformers import CrossEncoder
import gradio as gr

from neosophia.llmtools import openaiapi as openai

import neosophia.search.embeddings as emb
import neosophia.search.utils.doctree as doctree
from neosophia.search.utils import tree
from neosophia.search.semantic_search import cosine_search, rerank
from neosophia.search.utils.doctree import flatten_doctree
from neosophia.search.utils.data_utils import get_dict_hash

from examples import project

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Basic setup ===================================================
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
    '2023 FIA Formula One Financial Regulations': 'fia_formula_1_financial_regulations_-_issue_16_-_2023-08-31.yaml'
}
MEMO_DIRS = {
    '2023 FIA Formula One Sporting Regulations': 'sporting',
    '2023 FIA Formula One Financial Regulations': 'financial'
}


def load_memos() -> dict[str, list]:
    """Load certain sections of "memos" that represent a fictional team's policies."""
    bullet_p = r'^(\d\.)'
    article_p = r'\[(\d+\.\d.*?)\]'

    memos = {key:[] for key in REGS}
    memo_inds = {
        '2023 FIA Formula One Sporting Regulations': [22,24,26,28,31,43,53,54],
        '2023 FIA Formula One Financial Regulations': [i for i in range(1,12) if i!=10]
    }
    for key in REGS:
        if key not in memo_inds:
            continue
        for i in memo_inds[key]:
            doc_name = f'Memo_section_{i}.txt'
            with open(DOC_DIR / 'memos' / MEMO_DIRS[key] / doc_name, 'r') as f:
                file_lines = f.readlines()

            for line in file_lines:
                m = re.match(bullet_p, line)
                if m is not None:
                    bullet = m[1]
                    articles = re.findall(article_p, line)
                    if articles is not None:
                        new_line = line[:]
                        for article in articles:
                            new_line = new_line.replace(f'[{article}]', '')
                        new_line = new_line.replace(bullet, '').strip()
                        memos[key].append({
                            'text': new_line,
                            'articles': articles,
                            'document': doc_name
                        })

    return memos


def load_regs() -> dict[str, doctree.DocTree]:
    """Get regulations as doc trees."""
    doc_trees = {}
    for reg,filename in REGS.items():
        doc_trees[reg] = doctree.read(DOC_DIR / filename)

    return doc_trees


def load_defs() -> dict[str, list]:
    defs = {}
    for reg,filename in REGS.items():
        filename = DOC_DIR / filename.replace('yaml', 'defs')
        if filename.exists():
            with open(filename, 'r') as f:
                defs[reg] = yaml.safe_load(f)

    with open(DOC_DIR / 'formula_one_glossary.defs', 'r') as f:
        defs['F1 Glossary'] = yaml.safe_load(f)

    return defs


def encode(run_dir, filename, flat_texts, model):
    # Generate/retrieve embeddings
    filename = run_dir / filename
    if os.path.isfile(filename):
        log.info('Loading embeddings')
        embeddings = emb.load_embeddings(filename)
        log.info('Done.')
    else:
        log.info('Generating embeddings (this will take a few minutes)')
        embeddings = emb.encode(flat_texts, model)
        emb.save_embeddings(embeddings, filename)
        log.info('Done.')

    return embeddings


def get_embeddings(doc_trees, run_dir, model) -> tuple[torch.Tensor, list[str], list[tuple]]:
    """Flatten the texts for embedding, expanding per config"""
    flat_texts = []
    flat_ids = []
    if PRE_EXPAND:
        log.info('Expanding context window')
        for reg,doc_tree in doc_trees.items():
            ids, chunks = [], []
            for ind,item in flatten_doctree(doc_tree):
                tree_ind = ind[:-1]
                expanded_items = doctree.expand(item, doc_tree, tree_ind)
                chunks += expanded_items
                ids += [ind]*len(expanded_items)
            flat_texts += chunks
            flat_ids += [(reg,)+id for id in ids]
    else:
        for reg,doc_tree in doc_trees.items():
            ids,chunks = zip(*flatten_doctree(doc_tree))
            flat_texts += chunks
            flat_ids += [(reg,)+id for id in ids]

    embeddings = encode(run_dir, 'embeddings.pkl', flat_texts, model)

    return embeddings, flat_texts, flat_ids


def result_to_string(result: dict, doc_trees: list[doctree.DocTree]) -> str:
    file = result['file']
    section_ind = result['tree_index']

    section = tree.get_from_tree(doc_trees[file], section_ind)
    text = result['text']

    top_ind = section_ind
    section_headings = [section.title,]
    while (next_up:=tree.move_up(top_ind)) is not None:
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

    uri = DOC_DIR / f'{section.metadata["file"]}'
    uri = uri.absolute().as_uri() + f'#page={section.metadata["page"]}'


    summary_str = (
        f'{file.title()} Regulation: {", ".join(section_headings)}\n\n'
        f'{text}\n'
        f'- Similarity Score = {result["similarity_score"]}\n\n'
        f'- Re-ranked score = {result["reranked_score"]}\n\n'
        f'- `URI: {uri}`'
    )

    return summary_str


def results_to_string(results: list, doc_trees: list[doctree.DocTree]) -> str:
    output = [result_to_string(result, doc_trees) for result in results]

    return '\n\n---\n\n'.join(output)


def run_demo():
    """Run demo."""

    api_key = openai.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    openai.set_api_key(api_key)

    # Get a model
    llm_model = openai.start_chat('gpt-4')
    system_message = (
        'You are an assitant to a Formula 1 team.  Your job is to answer team questions '
        'to the best of your ability.  You will be given several excerpts from regulations '
        'to help you in answering.  Some of the provided regulations may be irrelevant, so '
        'ignore these and only use those that appear to be relevant. Do not mention or cite '
        'regulations that are not given to you.  Answer succinctly and make reference to the '
        'relevant regulation sections.\n\n'
        'Think carefully about the question and the provided regulations. Check that your '
        'response makes sense before answering.'
    )

    log.info('Loading memos')
    memo_set = load_memos()
    memo_texts = [
        f'{memo_type.upper()}:: {memo["text"]}'
        for memo_type,memo_list in memo_set.items()
        for memo in memo_list
    ]

    model_name = 'all-mpnet-base-v2'
    cross_encoder_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
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
        run_dir.mkdir()
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(config, f)

    doc_trees = load_regs()

    model = emb.get_model(model_name)
    embeddings, flat_texts, flat_ids = get_embeddings(doc_trees, run_dir, model)
    if RERANK:
        rerank_model = CrossEncoder(cross_encoder_name)

    log.info(f'Embeddings -- {type(embeddings)} -- {embeddings.shape}')

    # Get definitions
    definitions = load_defs()
    definitions_flat = []
    definition_ids = []
    for filename,defs in definitions.items():
        definitions_flat += defs
        definition_ids += [(filename,0)]*len(defs)

    definition_embeddings = encode(run_dir, 'embeddings_definitions.pkl', definitions_flat, model)
    del definitions

    def search_definitions(query_emb) -> str:
        results = cosine_search(query_emb, definition_embeddings, definition_ids, definitions_flat, model, 2)
        return '\n\n'.join([f'{result["text"]} (score={result["similarity_score"]}, from {result["file"]})' for result in results])

    def search(query: str) -> str:
        query_emb = emb.encode(query, model)
        definitions = search_definitions(query_emb)

        results = cosine_search(query_emb, embeddings, flat_ids, flat_texts, model, top_k)
        if RERANK:
            results = rerank(results, doc_trees, query, rerank_model, post_expand=POST_EXPAND)

        string_results = results_to_string(results[:5], doc_trees)

        context = (
            'Here are some potentially useful definitions:\n\n'
            f'{definitions}\n\n'
            '---\n\n'
            'Here is potentially useful context from the regulations:\n\n'
            f'{string_results}'
        )
        return context

    def generate_response(question:str, context: str) -> str:
        prompt = context + f'\n\nHere is the question: {question}'
        messages = [
            openai.Message('system', system_message),
            openai.Message('user', prompt)
        ]
        # return llm_model(messages).content
        return 'Skipping the LLM part'

    with gr.Blocks() as demo:
        gr.Markdown('# FIA Regulation Matching')
        policy_selector = gr.Dropdown(memo_texts, label='Select a memorandum item:')
        policy_text = gr.Textbox('Memo item or question:', label='Search box', interactive=True)
        search_button = gr.Button('Search')
        llm_response = gr.Textbox('LLM Response', interactive=False)
        output_text = gr.Markdown('Output will appear here')

        policy_selector.change(
            fn=lambda inp: inp.split('::')[-1],
            inputs=policy_selector,
            outputs=policy_text
        )

        search_button.click(
            fn=search,
            inputs=policy_text,
            outputs=output_text
        ).then(
            fn=generate_response,
            inputs=[policy_text, output_text],
            outputs=llm_response
        )

    demo.queue()
    demo.launch()


if __name__=='__main__':
    run_demo()