"""
Very basic semantic search example. Mainly used for verifying dependencies.
We can get rid of this when we have a more relevant text search example.

Expects the `hammurabi_openai.hf` dataset to be in the dataset directory.
"""

# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Ben Zimmer

from typing import Dict, List
import os

import readline  # replaces `input` with an improved version
import datasets as hfd
import numpy as np
import torch
import tqdm

from neosophia.llmtools import openaiapi as oaiapi

from examples import project

OPENAI_LLM_MODEL_NAME = 'gpt-3.5-turbo'
MAX_RULES = 3
QUIT_KEYWORDS = ['q', 'quit', 'x', 'exit']


def main() -> int:
    """main program"""

    # configure stuff
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    oaiapi.set_api_key(api_key)

    # load rules and embeddings from HFD into a simple list of dictionaries
    # with fields including "name", "text" and the torch tensor embedding "emb"
    rules_hfd = hfd.load_from_disk(
        os.path.join(
            project.DATASETS_DIR_PATH, 'hammurabi', 'hammurabi_openai.hf'))
    rules = list(rules_hfd)

    while True:

        search_str = input('Search string >')
        if not search_str or search_str in QUIT_KEYWORDS:
            return 0

        # get embedding of search string from OpenAI
        search_emb = oaiapi.extract_embeddings(oaiapi.embeddings([search_str]))[0]

        # perform a very simple vector search
        rule_idxs = find_most_similar_idxs(rules, search_emb, MAX_RULES)

        for idx in rule_idxs:
            print(rules[idx]['name'])
            print(rules[idx]['text'])
            print('~~~~ ~~~~ ~~~~ ~~~~')
        print()


def find_most_similar_idxs(records: List[Dict], emb: torch.Tensor, n: float) -> List[int]:
    """Simplest vector search implementation that performs a linear search."""
    scores = []
    for idx, record in tqdm.tqdm(enumerate(records)):
        score = torch.sum((emb - record['emb']) ** 2)
        scores.append(score.item())
    return np.argsort(scores)[:n]


if __name__ == '__main__':
    main()
