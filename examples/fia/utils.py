import os
import re
from pathlib import Path
import yaml
import logging

import torch

import neosophia.search.utils.doctree as doctree
import neosophia.search.embeddings as emb

log = logging.getLogger('search')

def load_memos(regulations: dict, memos_dirs: dict, doc_dir: Path) -> list[str]:
    """Load certain sections of "memos" that represent a fictional team's policies."""
    bullet_p = r'^(\d\.)'
    article_p = r'\[(\d+\.\d.*?)\]'

    memos = {key:[] for key in regulations}
    memo_inds = {
        '2023 FIA Formula One Sporting Regulations': [22,24,26,28,31,43,53,54],
        '2023 FIA Formula One Financial Regulations': [i for i in range(1,12) if i!=10]
    }
    for key in regulations:
        if key not in memo_inds:
            continue
        for i in memo_inds[key]:
            doc_name = f'Memo_section_{i}.txt'
            with open(doc_dir / 'memos' / memos_dirs[key] / doc_name, 'r') as f:
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


def load_regs(regulations: dict, doc_dir: Path) -> dict[str, doctree.DocTree]:
    """Get regulations as doc trees."""
    doc_trees = {}
    for reg,filename in regulations.items():
        doc_trees[reg] = doctree.read(doc_dir / filename)

    return doc_trees


def load_defs(regulations: dict, doc_dir: Path) -> tuple[list[str], list[tuple]]:
    """Get flat definition lists.

    Definition ids are simply file references, but they use the same indexing format
    as the DocTrees for consistency in retrieval."""
    defs = {}
    for reg,filename in regulations.items():
        filename = doc_dir / filename.replace('yaml', 'defs')
        if filename.exists():
            with open(filename, 'r') as f:
                defs[reg] = yaml.safe_load(f)

    with open(doc_dir / 'formula_one_glossary.defs', 'r') as f:
        defs['F1 Glossary'] = yaml.safe_load(f)

    definitions_flat = []
    definition_ids = []
    for filename,defs in defs.items():
        definitions_flat += defs
        definition_ids += [(filename,0)]*len(defs)

    return definitions_flat, definition_ids


def encode(run_dir: Path, filename: str, flat_texts: list[str], model) -> torch.Tensor:
    """Load or generate embeddings."""

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


def get_embeddings(
        doc_trees: dict[str, doctree.DocTree],
        run_dir: Path, model,
        pre_expand: bool
    ) -> tuple[torch.Tensor, list[str], list[tuple]]:
    """Flatten the texts for embedding, expanding per config"""
    flat_texts = []
    flat_ids = []
    if pre_expand:
        log.info('Expanding context window')
        for reg,doc_tree in doc_trees.items():
            ids, chunks = [], []
            for ind,item in doctree.flatten_doctree(doc_tree):
                tree_ind = ind[:-1]
                expanded_items = doctree.expand(item, doc_tree, tree_ind)
                chunks += expanded_items
                ids += [ind]*len(expanded_items)
            flat_texts += chunks
            flat_ids += [(reg,)+i for i in ids]
    else:
        for reg,doc_tree in doc_trees.items():
            ids,chunks = zip(*doctree.flatten_doctree(doc_tree))
            flat_texts += chunks
            flat_ids += [(reg,)+id for id in ids]

    embeddings = encode(run_dir, 'embeddings.pkl', flat_texts, model)

    return embeddings, flat_texts, flat_ids
