"""
Interactive application to demonstrate rewriting text to a template using
Basis of Presentation paragraphs from 10-Qs.
"""
import os
import json
import random
import difflib

from typing import List

import gradio as gr
import datasets as hfd

from examples import project
from neosophia.llmtools import openaiapi as oaiapi

opj = os.path.join

OPENAI_LLM_MODEL_NAME = 'gpt-4'


def generate_output(base_prompt: str, context: str, template: str) -> str:
    """Rewrite text to match a template."""
    base_prompt += 'Template: ' + template + '\n\n---------------------------\n\n'
    prompt = base_prompt + 'Input Text: ' + context
    return oaiapi.chat_completion(
        prompt=prompt,
        model=OPENAI_LLM_MODEL_NAME)


def extract_basis_data(dataset: hfd.Dataset) -> List[str]:
    """
    Use some heuristics to extract Basis of Presentation paragraphs
    from a dataset of 10-Q sections.
    """

    basis_data = []
    for item in dataset:
        if 'Table of Contents' in item['section']:
            continue
        if item['ticker'] == 'ACAN':
            continue
        if 'Item 1' not in item['section_id']:
            continue
        if 'Basis of Presentation' in item['section']:
            idx = item['section'].index('Basis of Presentation')
            text = item['section'][idx:]
            _text = text.split('\n\n')
            if 'Overview' in _text[0]:
                continue
            text = []
            for t in _text[1:]:
                if len(t) < 100:
                    break
                text.append(t)
            text = '\n\n'.join(text)
            if text == '':
                continue
            basis_data.append(text)

    return basis_data


def main():
    """Main program."""

    random.seed(0)

    # configure stuff
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH).rstrip()
    oaiapi.set_api_key(api_key)

    hf_file = opj(project.DATASETS_DIR_PATH, 'sec_10q_sections.hf')
    json_file = opj(project.DATASETS_DIR_PATH, 'sec_10q_sections.json')

    with open(opj(project.DATASETS_DIR_PATH, 'basis_template.txt'), 'r') as f:
        template = f.readlines()[0].rstrip()

    with open(
            opj(project.DATASETS_DIR_PATH, 'prompt_instructions.txt'), 'r') as f:
        base_prompt = f.readlines()[0].rstrip()

    if not os.path.exists(json_file):
        print('Generating json file...')
        dataset = hfd.load_from_disk(hf_file)
        basis_data = extract_basis_data(dataset)

        with open(json_file, 'w') as f:
            json.dump(basis_data, f, indent=4)
    else:
        print('Loading json file...')
        with open(json_file, 'r') as f:
            basis_data = json.load(f)

    random.shuffle(basis_data)

    idx = [10]
    def next_filing():
        """select the next filing from the dataset"""
        idx[0] = idx[0] + 1
        context = basis_data[idx[0]]
        output = generate_output(base_prompt, context, template)
        return context, output

    print('\nInitiating first rewrite...')
    initial_basis, initial_rev = next_filing()

    with gr.Blocks() as demo:
        gr.Markdown('# SEC Filing Language Standardizer')
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label='Original Basis of Presentation', value=initial_basis)
            with gr.Column():
                text_output = gr.Textbox(
                    label='Revised Basis of Presentation', value=initial_rev)
        with gr.Row():
            with gr.Column():
                next_button = gr.Button('Next')

        next_button.click(
            next_filing, inputs=None, outputs=[text_input, text_output])

    demo.launch()


if __name__ == '__main__':
    main()
