"""
Interactive application to demonstrate rewriting text to a template using
Basis of Presentation paragraphs from 10-Qs.
"""
import os
import json
import random

from typing import List, Tuple

import click
import gradio as gr
import datasets as hfd

from examples import project
from neosophia.llmtools import openaiapi as oaiapi

opj = os.path.join

OPENAI_LLM_MODEL_NAME = 'gpt-4'


def generate_output(base_prompt: str, context: str, template: str) -> str:
    """Rewrite text to match a template."""
    base_prompt += '\n\nTemplate: ' + template + '\n\n---------------------------\n\n'
    prompt = base_prompt + 'Input Text: ' + context
    return oaiapi.chat_completion(
        prompt=prompt,
        model=OPENAI_LLM_MODEL_NAME)


def extract_basis_data(dataset: hfd.Dataset) -> List[Tuple[str, str]]:
    """
    Use some heuristics to extract Basis of Presentation paragraphs
    from a dataset of 10-Q sections.
    """

    basis_data = []
    for item in dataset:
        if 'Table of Contents' in item['section']:
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

            # This assumes only one basis of presentation section per file
            uid = item['ticker'] + '-' + item['cik'] + '-' + item['date']
            basis_data.append((text, uid))

    return basis_data


@click.command()
@click.option(
    '--template_file', '-t',
    default=f"{opj(project.DATASETS_DIR_PATH, 'basis_template.txt')}")
@click.option(
    '--prompt_file', '-p',
    default=f"{opj(project.DATASETS_DIR_PATH, 'prompt_instructions.txt')}")
def main(template_file, prompt_file):
    """Main program."""

    random.seed(0)

    # configure stuff
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH).rstrip()
    oaiapi.set_api_key(api_key)

    hf_file = opj(project.DATASETS_DIR_PATH, 'sec_10q_sections.hf')
    json_file = opj(project.DATASETS_DIR_PATH, 'sec_10q_sections.json')

    if not os.path.exists(template_file):
        print(f'\nFile {template_file} does not exist\n')
        exit()
    if not os.path.exists(prompt_file):
        print(f'\nFile {prompt_file} does not exist\n')
        exit()

    with open(template_file, 'r') as f:
        template = ''.join(f.readlines())

    with open(prompt_file, 'r') as f:
        base_prompt = ''.join(f.readlines())

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
        res = basis_data[idx[0]]
        context, uid = res
        output = generate_output(base_prompt, context, template)
        with open(opj(project.DATASETS_DIR_PATH, uid + '.txt'), 'w') as f:
            f.write(output)
        print('Saved to', opj(project.DATASETS_DIR_PATH, uid + '.txt'))
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
                    label='Revised Basis of Presentation', value=initial_rev, interactive=True)
        with gr.Row():
            with gr.Column():
                next_button = gr.Button('Next')

        next_button.click(
            next_filing, inputs=None, outputs=[text_input, text_output])

    demo.launch()


if __name__ == '__main__':
    main()
