"""
Very basic semantic search example. Mainly used for verifying dependencies.
We can get rid of this when we have a more relevant text to search.

Expects the `hammurabi_openai.hf` dataset to be in the dataset directory.
"""

# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Ben Zimmer
import os
import pickle
import readline  # replaces `input` with an improved version

from typing import Dict, List

import tqdm
import torch
import numpy as np
import datasets as hfd
import langchain
import langchain.llms

import neosophia.llmtools.util as util

from examples import project
from neosophia.llmtools import openaiapi as oaiapi

MAX_RULES = 3
QUIT_KEYWORDS = ['q', 'quit', 'x', 'exit']
OPENAI_LLM_MODEL_NAME = 'gpt-4'


def build_qa_chain(
        llm: langchain.llms.BaseLLM,
        verbose: bool
        ) -> langchain.LLMChain:
    """build the LLMChain for answering questions given countext"""

    # prompt based on langchain.stuff_prompt

    prompt = langchain.PromptTemplate(
        template=(
            'Use the following pieces of context to answer the question at the end. ' +
            # "If you don't know the answer, just say that you don't know,
            # don't try to make up an answer. ' +
            '\n\n{context}\n\n' +
            'Question: {question}\n' +
            'Helpful Answer:'
        ),
        input_variables=['context', 'question']
    )

    return langchain.LLMChain(
        prompt=prompt,
        llm=llm,
        verbose=verbose
    )


def main() -> int:
    """main program"""

    # configure stuff
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)

    with open('embeddings.pkl', 'rb') as f:
        records = pickle.load(f)

    rules = [
        {
            'name': str(x['rule_name']) + ' ' + str(x['section_label']),
            'text': x['text'],
            'emb': x['emb']
        } for x in records
    ]

    openai_llm = langchain.OpenAI(
        openai_api_key=api_key,
        model_name=OPENAI_LLM_MODEL_NAME
    )

    qa_chain = build_qa_chain(llm=openai_llm, verbose=True)

    while True:

        search_str = input('Search string > ')
        if not search_str or search_str in QUIT_KEYWORDS:
            return 0

        # get embedding of search string from OpenAI
        search_emb = oaiapi.extract_embeddings(
            oaiapi.embeddings([search_str]))[0]

        # perform a very simple vector search
        rule_idxs = find_most_similar_idxs(rules, search_emb, MAX_RULES)

        # find the rule_text and create context
        rule_text = [rules[idx]['text'] for idx in rule_idxs]
        context = '\n\n'.join(rule_text)

        # ask the question and get an answer
        answer = qa_chain.run(context=context, question=question)

        #for idx in rule_idxs:
        #    print(rules[idx]['name'])
        #    print(rules[idx]['text'])
        #    print('~~~~ ~~~~ ~~~~ ~~~~')
        print('answer:', answer)
        #print()


def find_most_similar_idxs(records: List[Dict], emb: torch.Tensor, n: float) -> List[int]:
    """Simplest vector search implementation that performs a linear search."""
    scores = []
    for idx, record in tqdm.tqdm(enumerate(records)):
        score = torch.sum((emb - record['emb']) ** 2)
        scores.append(score.item())
    return np.argsort(scores)[:n]



if __name__ == '__main__':
    main()
