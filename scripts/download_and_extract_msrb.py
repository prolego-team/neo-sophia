""" Simple script for downloading and extracting rules from MSRB """
# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Cameron Fabbri
import os
import re

from typing import Any, Dict, List
from dataclasses import dataclass

import fitz
import tqdm
import click

from datasets import Dataset

import examples.project as project
import neosophia.llmtools.util as util
import neosophia.llmtools.pdf_utils as pu

from neosophia.llmtools import openaiapi as oaiapi

opj = os.path.join


def get_input():
    a = input()
    if a == 'q':
        exit()


class Colors:
    # Define ANSI escape sequences for different colors
    RED = "\033[31m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    BLACK = "\033[30m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    WHITE = "\033[37m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"


def colorize(text, color):
    return f"{color}{text}{Colors.RESET}"


@dataclass
class Rule:
    uid: str
    description: str
    sections: List[Any]
    interpretations: List[Any]
    amendments: List[Any]

    def __str__(self):
        a = colorize('uid: ', Colors.GREEN) + self.uid + '\n'
        b = colorize('description: ', Colors.GREEN) + self.description + '\n'
        c = colorize('sections: ', Colors.GREEN)
        for s in self.sections:
            c += s + '\n'
        d = colorize('interpretations: ', Colors.GREEN)
        for s in self.interpretations:
            d += s + '\n'
        e = colorize('amendments: ', Colors.GREEN)
        for s in self.amendments:
            e += s + '\n'

        return a + b + c + d + e


def get_all_rule_ids(data: List[Dict], xy=(491, 771), d=5):

    rule_ids = []
    for line in data:
        x, y = line['origin']

        if x - d <= xy[0] <= x + d and y - d <= xy[1] <= y + d:
            rule_id = line['text'].replace('|', '').strip()
            if rule_id not in rule_ids:
                rule_ids.append(rule_id)

    return rule_ids

def is_within_bounds(bbox, data_bounds):
    bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = bbox
    data_min_x, data_min_y, data_max_x, data_max_y = data_bounds

    if (data_min_x <= bbox_min_x <= data_max_x) and (data_min_y <= bbox_min_y <= data_max_y) \
            and (data_min_x <= bbox_max_x <= data_max_x) and (data_min_y <= bbox_max_y <= data_max_y):
        return True
    else:
        return False


def is_rule_uid(data, rule_pattern, all_rule_ids):
    text = data['text']
    font = data['font']
    if re.match(rule_pattern, text) and text in all_rule_ids and 'Bold' in font:
        return True
    return False


def extract_msrb_rules(pdf, start_page, end_page):

    all_data = []
    pages = []
    blocks = []
    for page_num, page in enumerate(pdf):
        if page_num < start_page - 1:
            continue
        if page_num == end_page:
            break
        page_dict = page.get_text('dict')
        pages.append(page_dict)

    for page_dict in pages:
        blocks = page_dict['blocks']
        for block in blocks:
            if 'lines' in block.keys():
                for span in block['lines']:
                    data = span['spans']
                    for lines in data:
                        all_data.append(lines)

    all_rule_ids = get_all_rule_ids(all_data)

    # Bounding box of the actual content we want
    # (min_x, min_y, max_x, max_y)
    data_bounds = (70, 67, 587, 735)

    rule_pattern = r"Rule [A-Z]-\d+"
    section_pattern = r"^\([a-zA-z0-9]\)"

    state_other = -1
    state_uid = 0
    state_description = 1
    state_sections = 2
    state_interpretations = 3
    state_amendment = 4

    current_rule_uid = None

    rules = []

    state = state_other
    for page_num, page_dict in enumerate(pages):
        page_data = [lines
            for block in page_dict['blocks']
            if 'lines' in block.keys()
            for span in block['lines']
            for lines in span['spans']
        ]

        title = []
        for idx, data in enumerate(page_data):
            if not is_within_bounds(data['bbox'], data_bounds):
                continue
            text = data['text']

            if state == state_other:
                if is_rule_uid(data, rule_pattern, all_rule_ids):
                    state = state_uid
            elif state == state_uid:
                state = state_description
            elif state == state_description:
                if re.match(section_pattern, text):
                    state = state_sections
            elif state == state_sections:
                if text == current_rule_uid + ' Interpretations':
                    state = state_interpretations
                elif is_rule_uid(data, rule_pattern, all_rule_ids):
                    state = state_uid
            elif state == state_interpretations:
                if text.startswith(current_rule_uid + ' Amendment History'):
                    state = state_amendment
                elif is_rule_uid(data, rule_pattern, all_rule_ids):
                    state = state_uid
            elif state == state_amendment:
                if is_rule_uid(data, rule_pattern, all_rule_ids):
                    state = state_uid

            if state == state_uid:
                if current_rule_uid is not None:
                    rule_description = [x.strip() for x in rule_description]
                    rule = Rule(
                        uid=current_rule_uid,
                        description=' '.join(rule_description).strip(),
                        sections=rule_sections,
                        interpretations=rule_interpretations,
                        amendments=rule_amendments)
                    rules.append(rule)
                    print(rule)
                    exit()
                current_rule_uid = text
                rule_description = []
                rule_sections = []
                rule_interpretations = []
                rule_amendments = []
            elif state == state_description:
                rule_description.append(text)
            elif state == state_sections:
                rule_sections.append(text)
            elif state == state_interpretations:
                rule_interpretations.append(text)
            elif state == state_amendment:
                rule_amendments.append(text)

    if current_rule_uid is not None:
        rule_description = [x.strip() for x in rule_description]
        rule = Rule(
            uid=current_rule_uid,
            description=' '.join(rule_description).strip(),
            sections=rule_sections,
            interpretations=rule_interpretations,
            amendments=rule_amendments)
        rules.append(rule)

    exit()


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

    pdf = fitz.open(file_in)
    extract_msrb_rules(pdf, start_page, end_page)

    exit()
    sections, num_pages = pu.extract_text_by_paragraphs(
        file_in, delimiter, start_page, end_page)

    if end_page == -1:
        end_page = num_pages

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

    file_out = opj(
        project.DATASETS_DIR_PATH, f'MSRB-{start_page}-{end_page}.hfd')
    dataset= Dataset.from_dict({'records': records})
    dataset.save_to_disk(opj(project.DATASETS_DIR_PATH, file_out))


if __name__ == '__main__':
    main()

