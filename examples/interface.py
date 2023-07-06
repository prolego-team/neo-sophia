"""
"""
import os
import pickle
import textwrap

from typing import Dict, List

import tqdm
import torch
import numpy as np
import gradio as gr
import datasets as hfd
import langchain
import langchain.llms

import neosophia.llmtools.util as util

from examples import project
from neosophia.llmtools import openaiapi as oaiapi


def setup():
    """ Configuration and data loading """

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

    return records, rules


def find_most_similar_idxs(
        records: List[Dict], emb: torch.Tensor, n: float) -> List[int]:
    """Simplest vector search implementation that performs a linear search."""
    scores = []
    for idx, record in tqdm.tqdm(enumerate(records)):
        score = torch.sum((emb - record['emb']) ** 2)
        scores.append(score.item())
    return np.argsort(scores)[:n]


def rule_lookup(prompt):

    # get embedding of search string from OpenAI
    search_emb = oaiapi.extract_embeddings(
        oaiapi.embeddings([prompt]))[0]

    # perform a very simple vector search
    rule_idxs = find_most_similar_idxs(rules, search_emb, MAX_RULES)

    # find the rule_text and create context
    rule_text = [rules[idx]['text'] for idx in rule_idxs]
    context = '\n\n'.join(rule_text)

    # ask the question and get an answer
    answer = qa_chain.run(context=context, question=question)

    print('answer:', answer)


def build_rules_html(records, rules: List) -> str:

    rule_dict = {}

    for r in records:

        rule_name = r['rule_name']
        section_label = r['section_label']
        text = r['text']

        if rule_name not in rule_dict:
            rule_dict[rule_name] = {}

        if section_label not in rule_dict[rule_name]:
            rule_dict[rule_name][section_label] = text

    html = '<html>\n<body>'
    for rule_name, rule_sections in rule_dict.items():
        indent_level = 0
        html += f'<h2>{rule_name}</h2><br>'
        for section_labels, text in rule_sections.items():
            if None not in section_labels:
                indent_level = len(section_labels)
                section_labels = list(section_labels)[-1]
            else:
                section_labels = ''

            indent = ''.join(['  '] * indent_level)

            text = '\n'.join(textwrap.wrap(text, subsequent_indent=indent))
            html += f'<pre>{indent}<b>{section_labels})</b></pre> '
            html += f'<pre>{indent}{text}</pre>'

    html += '</body>\n</html>'
    with open('rules.html', 'w') as f:
        f.write(html)

    return html


def main():

    records, rules = setup()
    rules_html = build_rules_html(records, rules)
    exit()

    with gr.Blocks() as demo:
        gr.Markdown("# Welcome")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox()
                text_button = gr.Button('Search')
            with gr.Column():
                text_output = gr.Textbox()

        text_button.click(rule_lookup, inputs=text_input, outputs=text_output)

    demo.launch()


if __name__ == '__main__':
    main()

