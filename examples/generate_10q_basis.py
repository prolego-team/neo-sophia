"""
"""

# Cameron Fabbri + Ben Zimmer
import os
import json
import random
import difflib

from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from datasets import load_from_disk

from examples import project
from neosophia.llmtools import openaiapi as oaiapi

opj = os.path.join

OPENAI_LLM_MODEL_NAME = 'gpt-4'


def color_diff(str1, str2):

    diff = difflib.ndiff(str1.splitlines(), str2.splitlines())
    output = ""
    for line in diff:
        if line.startswith('-'):
            # Red color for deleted lines
            output += f"\033[31m{line}\033[0m\n"
        elif line.startswith('+'):
            # Green color for added lines
            output += f"\033[32m{line}\033[0m\n"
        elif line.startswith('?'):
            # Cyan color for modified lines
            output += f"\033[36m{line}\033[0m\n"
        else:
            # Unchanged lines
            output += f"{line}\n"

    return output


def generate_output(context: str, template: str) -> List[str]:
    base_prompt = 'Rewrite the following text to conform to the given template:\n'
    base_prompt += 'Template: ' + template + '\n\n---------------------------\n\n'
    prompt = base_prompt + 'Input Text: ' + context
    return oaiapi.chat_completion(
        prompt=prompt,
        model=OPENAI_LLM_MODEL_NAME)


def extract_basis_data(dataset):

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
    """main program"""

    # configure stuff
    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH).rstrip()
    oaiapi.set_api_key(api_key)

    hf_file = opj('data', 'sec_10q_sections.hf')
    json_file = opj('data', 'sec_10q_sections.json')

    if not os.path.exists(json_file):
        print('Generating json file...')
        dataset = load_from_disk(hf_file)
        basis_data = extract_basis_data(dataset)

        with open(json_file, 'w') as f:
            json.dump(basis_data, f, indent=4)
    else:
        print('Loading json file...')
        with open(json_file, 'r') as f:
            basis_data = json.load(f)

    random.shuffle(basis_data)

    def generate_template():
        prompt = ['Write a template given the following examples of "Basis of Presentation": ']

        for c in basis_data[:10]:
            prompt.append(
                c + '\n----------------------------------------------\n')

        prompt = ''.join(prompt)

        return oaiapi.chat_completion(
            prompt=prompt,
            model=OPENAI_LLM_MODEL_NAME
        )


    idx = [10]
    def next_filing():
        idx[0] = idx[0] + 1
        return basis_data[idx[0]]

    with gr.Blocks() as demo:
        gr.Markdown('# 10-Q Template Thingy')
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label='10-Q Basis', value=basis_data[idx[0]])
            with gr.Column():
                template = gr.Textbox(label='Template')
            with gr.Column():
                text_output = gr.Textbox(label='Rewritten 10-Q Basis')
        with gr.Row():
            with gr.Column():
                next_button = gr.Button('Next')
            with gr.Column():
                gen_template_button = gr.Button('Generate Template')
            with gr.Column():
                gen_output_button = gr.Button('Generate Output')
        #with gr.Row():
        #    diff = gr.HTML(label='Diff')
        #with gr.Row():
        #    show_diff_button = gr.Button('Show Diff')

        next_button.click(
            next_filing, inputs=None, outputs=text_input)
        gen_template_button.click(
                generate_template, inputs=None, outputs=template)
        gen_output_button.click(
            generate_output,
            inputs=[text_input, template],
            outputs=text_output)
        #show_diff_button.click(
        #    color_diff, inputs=[text_input, text_output], outputs=diff)

    demo.launch()


if __name__ == '__main__':
    main()
