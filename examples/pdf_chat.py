"""
"""
import os
import re
import time
import pickle

from collections import Counter

import click
import torch
import chromadb
import numpy as np
import gradio as gr

from tqdm import tqdm

import neosophia.db.chroma as chroma

from examples import project

from neosophia.llmtools import openaiapi as oaiapi, pdf_utils, text_utils
from neosophia.db.pdfdb import PDFDB


@click.command()
@click.option(
    '--data_dir', '-f', help='Path to a directory containing PDFs',
    default=f'{project.DATASETS_DIR_PATH}/Title-12')
def main(data_dir):
    """ main """

    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)

    # Create database
    pdfdb = PDFDB('pdf_db', api_key)

    # Add PDFs from the directory to the database
    filenames = sorted(pdf_utils.find_pdfs_in_directory(data_dir))
    for filename in tqdm(filenames):
        pdfdb.add_pdf(filename)

    base_prompt = 'Determine if the following context has enough information to answer the question. If it doesn\'t, say "There is not enough context to provide an answer". If it does, provide the answer. THINK STEP BY STEP. ALWAYS PROVIDE QUOTES AND PAGE CITATIONS. ALWAYS PROVIDE AN ESTIMATED CONFIDENCE SCORE WITH YOUR ANSWER\n\n'

    def ask_question(question):

        # Get the top n closest pages for the question
        result = pdfdb.get_context_from_query(question, n_results=4)

        answers = []
        for res in result:
            prompt = base_prompt
            prompt += 'Question: ' + question
            prompt += res

            page_number = re.search(r'\bPage Number: (\d+)\b', res).group(1)

            out = oaiapi.chat_completion(
                prompt=prompt,
                model='gpt-4')

            if 'There is no' in out:
                print(f'Not enough information on page {page_number}.')
                continue
            else:
                answers.append(out)

        return '\n--\n'.join(answers)

    question = 'What happens when a national bank\'s investment in securities no longer conforms to the regulations but conformed when made?'
    with gr.Blocks() as demo:
        gr.Markdown('# Semantic Search')
        with gr.Row():
            text_input = gr.Textbox(value=question)
        with gr.Row():
            question_button = gr.Button('Ask a question')

        gr.Markdown("# Results")
        with gr.Row():
            text_output = gr.Textbox()

        question_button.click(
            ask_question, inputs=text_input, outputs=text_output)

    demo.launch()


if __name__ == '__main__':
    main()

