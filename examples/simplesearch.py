"""
Very basic semantic search / question answering example.

"""

# Copyright (c) 2023 Prolego Inc. All rights reserved.
import os
import pickle
import readline  # replaces `input` with an improved version

from typing import Dict, List

import tqdm
import torch
import openai
import numpy as np

import neosophia.llmtools.util as util

from examples import project
from neosophia.llmtools import openaiapi as oaiapi

MAX_RULES = 5
QUIT_KEYWORDS = ['q', 'quit', 'x', 'exit']
OPENAI_LLM_MODEL_NAME = 'gpt-4'


def qa_func(context: str, question: str) -> str:
    prompt = (
            'Use the following pieces of context to answer the question at the end. Directly reference the rules used to determine your answer. ' +
        # "If you don't know the answer, just say that you don't know,
        # don't try to make up an answer. ' +
        f'\n\n{context}\n\n' +
        f'Question: {question}\n' +
        'Helpful Answer:'
    )

    return oaiapi.chat_completion(
        prompt=prompt,
        model=OPENAI_LLM_MODEL_NAME
    )


def main() -> int:
    """main program"""

    # configure stuff
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH).rstrip()
    oaiapi.set_api_key(api_key)

    with open('embeddings.pkl', 'rb') as f:
        records = pickle.load(f)

    rules = [
        {
            'name': str(x['rule_name']) + ' ' + str(x['section_label']),
            'text': x['text'],
            'emb': x['emb']
        } for x in records
    ]

    def _level(name: str) -> int:
        """find the level of the rule"""
        return len([x for x in name if x == '\'']) // 2

    # keep top level rules only
    rules = [
        rule for rule in rules
        if _level(rule['name']) == 1
    ]

    # for x in rules:
    #     print(x['name'])
    #     print('--------')

    while True:

        search_str = input('Question > ')
        if not search_str or search_str in QUIT_KEYWORDS:
            return 0

        # get embedding of search string from OpenAI
        search_emb = oaiapi.extract_embeddings(
            oaiapi.embeddings([search_str]))[0]

        # perform a very simple vector search
        rule_idxs = find_most_similar_idxs(rules, search_emb, MAX_RULES)

        print('potentially relevant rules:\n')
        for idx in rule_idxs:
            print(rules[idx]['name'])
            print(rules[idx]['text'])
            print('~~~~ ~~~~ ~~~~ ~~~~')

        # find the rule_text and create context
        rule_text = [rules[idx]['text'] for idx in rule_idxs]
        context = '\n\n'.join(rule_text)

        # ask the question and get an answer
        # answer = qa_chain.run(context=context, question=search_str)
        answer = qa_func(context=context, question=search_str)

        print('answer:', answer)
        print()


def find_most_similar_idxs(records: List[Dict], emb: torch.Tensor, n: float) -> List[int]:
    """Simplest vector search implementation that performs a linear search."""
    scores = []
    for idx, record in tqdm.tqdm(enumerate(records)):
        score = torch.sum((emb - record['emb']) ** 2)
        scores.append(score.item())
    return np.sort(scores)[:n], np.argsort(scores)[:n]


if __name__ == '__main__':
    main()
