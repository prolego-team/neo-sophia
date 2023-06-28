""" Simple script for downloading and extracting rules from MSRB """
# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Cameron Fabbri
import os

import fitz
import tqdm
import click

from datasets import Dataset

import examples.project as project
import neosophia.llmtools.util as util
import neosophia.llmtools.pdf_utils as pu

from neosophia.llmtools import openaiapi as oaiapi

opj = os.path.join

@click.command()
@click.option('--start_page', '-s', default=16)
@click.option('--end_page', '-e', default=-1)
@click.option('--delimiter', default=r"^\([a-zA-z0-9]\)")

def main(start_page, end_page, delimiter):
    """ """

    url = 'https://www.msrb.org/sites/default/files/MSRB-Rule-Book-Current-Version.pdf'
    filename = 'MSRB-Rule-Book-Current-Version.pdf'

    file_in = opj(project.DATASETS_DIR_PATH, filename)

    if not os.path.exists(file_in):
        print(f'Downloading MSRB to {project.DATASETS_DIR_PATH} ...')
        os.system(f'wget -P {project.DATASETS_DIR_PATH} {url}')
        print('Done')

    # If only extracting one page, still have to give a range
    if start_page == end_page:
        end_page += 1

    sections, num_pages = pu.extract_text_by_paragraphs(
        file_in, delimiter, start_page, end_page)

    if end_page == -1:
        end_page = num_pages

    file_out = opj(
        project.DATASETS_DIR_PATH, f'MSRB-Text-{start_page}-{end_page}.txt')

    '''
    with open(file_out, 'w') as f:
        for section in sections:
            section = ' '.join(section.strip().split('\n'))
            split_section = section.split(')')
            rule_id = split_section[0] + ')'
            rule_text = ' '.join(split_section[1:]).strip()
            f.write(f'{rule_id}~|~{rule_text}\n')
    print(f'Wrote {len(sections)} sections to {file_out}')
    '''

    lines = []
    for section in sections:
        section = ' '.join(section.strip().split('\n'))
        split_section = section.split(')')
        rule_id = split_section[0] + ')'
        rule_text = ' '.join(split_section[1:]).strip()
        lines.append([rule_id, rule_text])

    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)

    records = []
    idx = 0
    for line in tqdm.tqdm(lines):
        if line[1] == '':
            continue
        line[0] = line[0].rstrip()
        emb = oaiapi.extract_embeddings(oaiapi.embeddings(line[1]))[0]
        records.append(
            {
                'name': line[0],
                'text': line[1],
                'emb': emb
            }
        )

        idx += 1

    dataset= Dataset.from_dict({'records': records})
    dataset.save_to_disk(opj(project.DATASETS_DIR_PATH, 'MSRB.hfd'))


if __name__ == '__main__':
    main()

