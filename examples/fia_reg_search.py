import os
from pathlib import Path
import re
import json
import logging
import sys
from collections import Counter, defaultdict

import yaml
import torch
from sentence_transformers import CrossEncoder
import gradio as gr

from neosophia.llmtools import openaiapi as openai

import neosophia.search.embeddings as emb
import neosophia.search.utils.doctree as doctree
from neosophia.search.utils import tree
from neosophia.search.semantic_search import cosine_search, rerank
from neosophia.search.keyword_search import keyword_search, build_index
from neosophia.search.utils.doctree import flatten_doctree
from neosophia.search.utils.data_utils import get_dict_hash

from neosophia.agents.react_chat import make_react_agent

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

MAX_LLM_CALLS_PER_INTERACTION = 10

FUNCTION_DESCRIPTIONS = [
    {
        'name': 'lookup',
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
    }
]


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
    memo_texts = [
        f'{memo_type.upper()}:: {memo["text"]}'
        for memo_type,memo_list in memos.items()
        for memo in memo_list
    ]
    return memo_texts


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

    definitions_flat = []
    definition_ids = []
    for filename,defs in defs.items():
        definitions_flat += defs
        definition_ids += [(filename,0)]*len(defs)

    return definitions_flat, definition_ids


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
        # f'- Similarity Score = {result["similarity_score"]}\n\n'
        # f'- Re-ranked score = {result["reranked_score"]}\n\n'
        # f'- `URI: {uri}`'
    )

    return summary_str


def results_to_string(results: list, doc_trees: list[doctree.DocTree]) -> str:
    output = [result_to_string(result, doc_trees) for result in results]

    return '\n\n---\n\n'.join(output)


def run_demo():
    """Run demo."""

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
        run_dir.mkdir()
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
        'You can look up any word or phrase that you are not sure about using the lookup function.\n\n'
        'Think carefully about the question and the provided regulations. Check that your '
        'response makes sense before answering.'
    )

    # Load data
    log.info('Loading memos')
    memo_texts = load_memos()

    log.info('Loading regulations')
    doc_trees = load_regs()

    log.info('Loading definitions')
    definitions_flat, definition_ids = load_defs()

    # Get encodings
    log.info('Getting encodings for regs and definitions')
    embeddings, flat_texts, flat_ids = get_embeddings(doc_trees, run_dir, model)
    log.info(f'Embeddings -- {type(embeddings)} -- {embeddings.shape}')

    # definition_embeddings = encode(run_dir, 'embeddings_definitions.pkl', definitions_flat, model)
    definition_bm25 = build_index(definitions_flat)

    # Wrapper functions for search and generation
    def search_definitions(query) -> list[str]:
        """Do a keyword search over definitions."""
        results = keyword_search(definition_bm25, query, definition_ids, definitions_flat, 3)
        return [
            (result["similarity_score"], f'{result["text"]} (from {result["file"]})')
            for result in results
        ]

    def search(query: str) -> str:
        """Do a semantic search over embeddings.  Also return potentially relevant definitiosn."""
        query_emb = emb.encode(query, model)
        definitions = search_definitions(query)

        results = cosine_search(query_emb, embeddings, flat_ids, flat_texts, model, top_k)
        if RERANK:
            results = rerank(results, doc_trees, query, rerank_model, post_expand=POST_EXPAND)

        for result in results:
            new_definitions = search_definitions(result['text'])
            if RERANK:
                new_definitions = [(score*result['reranked_score']/10,defn) for score,defn in new_definitions]
            else:
                new_definitions = [(score*result['similarity_score'],defn) for score,defn in new_definitions]
            definitions += new_definitions

        # Consolidate definitions list
        # def_scores, definitions = zip(*definitions)
        aggregator = defaultdict(int)
        for def_score, definition in definitions:
            aggregator[definition] += def_score
        weighted_definitions = [(defn,score) for defn,score in aggregator.items()]
        weighted_definitions = sorted(
            weighted_definitions,
            key=lambda x: x[1],
            reverse=True
        )
        weighted_definitions, _ = zip(*weighted_definitions)
        definitions = "\n\n".join(weighted_definitions[:5])

        # Apply a threshold to results
        if RERANK:
            good_results = [result for result in results if result['reranked_score']>-2]
        else:
            good_results = [result for result in results if result['similarity_score']>0.3]
        string_results = results_to_string(good_results, doc_trees)

        context = (
            'Here are some potentially useful definitions:\n\n'
            f'{definitions}\n\n'
            '---\n\n'
            'Here is potentially useful context from the regulations:\n\n'
            f'{string_results}'
        )
        return context

    def generate_response(question: str, context: str) -> str:
        prompt = context + f'\n\nHere is the question: {question}'
        messages = [
            openai.Message('system', system_message),
            openai.Message('user', prompt)
        ]
        return llm_model(messages).content

    def agentic_search(question: str, context: str) -> str:
        if False:
            return 'Done'
        functions = {
            'lookup': search_definitions
        }
        agent = make_react_agent(
            system_message,
            llm_model,
            FUNCTION_DESCRIPTIONS,
            functions,
            MAX_LLM_CALLS_PER_INTERACTION,
            extra_context=context
        )

        prompt = question #+ '\n\n' + context
        for message in agent(prompt):
            if message.role=='user':
                status = 'User agent asked a question or provided feedback to the LLM.  Awaiting LLM response...'
                log.info('User mesage: ' + message.content)
            elif message.role=='function':
                status = 'A query was run against the database.  Awaiting LLM response...'
                log.info('Function call: ' + f'\n\n_<name={message.name}, function_call={message.function_call}>_\n')
            elif message.role=='assistant':
                if 'Final Answer:' in message.content:
                    status = 'The final answer has been determined.'
                else:
                    status = 'The assistant responded.  Awaiting LLM next response...'
                log.info('Assistant mesage: ' + message.content)
            # yield chat_history

        additional_prompts = generate_response(
            question+'\n\nWhat are several other questions that could be asked to get a more precise answer?',
            context
        )
        return (
            message.content +
            '\n\nHere are several other questions related to this question:\n\n' +
            additional_prompts
        )

    def multi_search(question: str, context: str) -> str:
        log.info(question)
        log.info(context)
        base_question = question
        base_response = generate_response(base_question, context)
        log.info('Generating queries')
        additional_prompts = generate_response(
            question+'\n\nWhat are several other questions that could be asked to get a more precise or complete answer?  Return the questions as a numbered list.',
            context
        )
        log.info(additional_prompts)
        questions = additional_prompts.split('\n')
        responses = [base_response, ]
        max_qs = 3
        for question in questions[:max_qs]:
            log.info('Generating new response')
            context = search(question)
            responses.append(generate_response(question, context))

        new_context = '\n\n'.join([f'Response {i+1}: {response}' for i, response in enumerate(responses)])
        log.info(new_context)
        log.info('Synthesizing responses')
        prompt = (
            'Please summarize these responses into a single consolidated response to the '
            f'original question, "{base_question}". Remember to include citations of '
            'regulations where applicable.'
        )
        log.info(prompt)
        final_response = generate_response(prompt, new_context)

        return final_response


    with gr.Blocks() as demo:
        gr.Markdown('# FIA Regulation Matching')
        policy_selector = gr.Dropdown(memo_texts, label='Select a memorandum item:')
        policy_text = gr.Textbox('Memo item or question:', label='Search box', interactive=True)
        with gr.Row():
            quick_search_button = gr.Button('Quick Search')
            search_summarize_button = gr.Button('Search and Summarize')
            deep_search_button = gr.Button('Deep Search')
        llm_response = gr.Textbox('LLM Response', interactive=False)
        output_text = gr.Textbox('Output will appear here', interactive=False)

        policy_selector.change(
            fn=lambda inp: 'Where is this discussed in the FIA regulations: ' + inp.split('::')[-1],
            inputs=policy_selector,
            outputs=policy_text
        )

        quick_search_button.click(
            fn=search,
            inputs=policy_text,
            outputs=output_text
        ).then(
            fn=lambda : 'No LLM response requested.',
            inputs=None,
            outputs=llm_response
        )

        search_summarize_button.click(
            fn=search,
            inputs=policy_text,
            outputs=output_text
        ).then(
            fn=generate_response,
            inputs=[policy_text, output_text],
            outputs=llm_response
        )

        deep_search_button.click(
            fn=search,
            inputs=policy_text,
            outputs=output_text
        ).then(
            fn=multi_search, #agentic_search, #generate_response,
            inputs=[policy_text, output_text],
            outputs=llm_response
        )

    demo.queue()
    demo.launch()


if __name__=='__main__':
    run_demo()