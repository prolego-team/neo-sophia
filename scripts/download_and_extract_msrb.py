""" Simple script for downloading and extracting rules from MSRB """
# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Cameron Fabbri
import os
import re
import pickle

from typing import Any, Dict, List

import fitz
import tqdm
import click
import requests

from datasets import Dataset

import examples.project as project
import neosophia.llmtools.util as util

from neosophia.llmtools import openaiapi as oaiapi
from neosophia.datasets.msrb import Rule

opj = os.path.join

STATE_OTHER = -1
STATE_UID = 0
STATE_DESCRIPTION = 1
STATE_SECTIONS = 2
STATE_INTERPRETATIONS = 3
STATE_AMENDMENT = 4
STATE_IF = 5

EXCLUDE_RULES = ['Rule G-29', 'Rule G-35', 'Rule G-36', 'Rule A-6', 'Rule A-11']


def parse_rule_sections(section):

    rule_dict = {}

    current_indent = 0
    current_x_loc = section[0][1][0]
    delta = 5
    section_pattern = r"^\([a-zA-z0-9]+\)"

    labels = [None] * 10
    for paragraph in section:

        p_text = paragraph[0].strip()

        x_loc = paragraph[1][0]

        if x_loc > 300:
            x_loc -= 267

        if x_loc < current_x_loc - delta:
            current_indent -= int((current_x_loc - x_loc) / 17.)
            current_x_loc = x_loc

        #print(paragraph)
        #print('x_loc:', x_loc)
        #print('current_x_loc:', current_x_loc)
        #print('current_indent:', current_indent)
        match = re.match(section_pattern, p_text)
        if match is not None:

            if x_loc > current_x_loc + delta:
                current_indent += int((x_loc - current_x_loc) / 17.)
                #print('updated current_indent:', current_indent)
                current_x_loc = x_loc

            span = match.span()
            label = p_text[span[0] + 1:span[1] - 1]
            text = p_text[span[1]:].strip()
            labels[current_indent] = label

        else:
            text = p_text

        for idx in range(current_indent + 1):
            key = tuple(labels[0:idx + 1])
            paragraphs = rule_dict.setdefault(key, [])
            paragraphs.append(text)

        #print('\n--------------------------------------------------\n')

    return {k: ' '.join(v) for k, v in rule_dict.items()}


def get_input():
    a = input()
    if a == 'q':
        exit()


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


def is_rule_uid(data, rule_pattern):
    text = data['text']
    font = data['font']
    if re.match(rule_pattern, text) and 'Bold' in font:
        if 'Rule D-' in text:
            return False
        if 'Rule A-' in text:
            return False
        if 'Rule IF-' in text:
            return False
        for x in EXCLUDE_RULES:
            if x in text:
                return False
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

    # Bounding box of the actual content we want
    # (min_x, min_y, max_x, max_y)
    data_bounds = (70, 67, 590, 735)

    rule_pattern = r"Rule [A-Z]-\d+$"
    if_pattern = r"IF-\d+$"
    section_pattern = r"^\([a-zA-z0-9]\)"

    current_rule_uid = None

    rules = []

    state = STATE_OTHER
    for page_num, page_dict in enumerate(pages):

        title = []
        for idx, block in enumerate(page_dict['blocks']):

            if 'lines' not in block:
                continue
            if 'spans' not in block['lines'][0]:
                continue

            data = block['lines'][0]['spans'][0]

            if not is_within_bounds(data['bbox'], data_bounds):
                continue

            text = ''
            text_lines = []
            for line in block['lines']:
                for span in line['spans']:
                    text += span['text']
                    text_lines.append(span['text'])

            if state == STATE_OTHER:
                if is_rule_uid(data, rule_pattern):
                    state = STATE_UID
            elif state == STATE_UID:
                if re.match(
                    section_pattern, text) or data['font'] == 'Times-Roman':
                    state = STATE_SECTIONS
            elif state == STATE_SECTIONS:
                if text.startswith(current_rule_uid + ' Interpretation'):
                    state = STATE_INTERPRETATIONS
                elif is_rule_uid(data, rule_pattern):
                    state = STATE_UID
            elif state == STATE_INTERPRETATIONS:
                if text.startswith(current_rule_uid + ' Amendment History'):
                    state = STATE_AMENDMENT
                elif is_rule_uid(data, rule_pattern):
                    state = STATE_UID
            elif state == STATE_AMENDMENT:
                if is_rule_uid(data, rule_pattern):
                    state = STATE_UID
                elif text == 'IF-1':
                    state = STATE_IF

            if state == STATE_UID:
                if current_rule_uid is not None:
                    rule_description = [x.strip() for x in rule_description]
                    rule = Rule(
                        uid=current_rule_uid,
                        description=' '.join(rule_description).strip(),
                        sections=parse_rule_sections(rule_sections),
                        interpretations=rule_interpretations,
                        amendments=rule_amendments)
                    rules.append(rule)
                current_rule_uid = text_lines[0]
                rule_description = text_lines[1:]
                rule_sections = []
                rule_interpretations = []
                rule_amendments = []
            elif state == STATE_SECTIONS:
                rule_sections.append((text, data['origin']))
            elif state == STATE_INTERPRETATIONS:
                rule_interpretations.append(text)
            elif state == STATE_AMENDMENT:
                rule_amendments.append(text)

    if current_rule_uid is not None:
        rule_description = [x.strip() for x in rule_description]
        rule = Rule(
            uid=current_rule_uid,
            description=' '.join(rule_description).strip(),
            sections=parse_rule_sections(rule_sections),
            interpretations=rule_interpretations,
            amendments=rule_amendments)
        rules.append(rule)

    return {rule.uid: rule for rule in rules}


@click.command()
@click.option('--start_page', '-s', default=16)
@click.option('--end_page', '-e', default=-1)
@click.option('--delimiter', default=r"^\([a-zA-z0-9]\)")
def main(
        start_page: str,
        end_page: str,
        delimiter: str):
    """ """

    url = 'https://www.msrb.org/sites/default/files/MSRB-Rule-Book-Current-Version.pdf'
    filename = 'MSRB-Rule-Book-Current-Version.pdf'

    file_in = opj(project.DATASETS_DIR_PATH, filename)

    if not os.path.exists(file_in):
        print(f'Downloading MSRB to {project.DATASETS_DIR_PATH} ...')
        # os.system(f'wget -P {project.DATASETS_DIR_PATH} {url}')
        req = requests.get(url)
        with open(os.path.join(project.DATASETS_DIR_PATH, filename), 'wb') as f:
            f.write(req.content)
        exit()
        print('Done')

    # If only extracting one page, still have to give a range
    if start_page == end_page:
        end_page += 1

    pdf = fitz.open(file_in)
    rule_dict = extract_msrb_rules(pdf, start_page, end_page)

    api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    oaiapi.set_api_key(api_key)

    print('Generating embeddings for rules...')
    records = []
    for rule_name, rule_section in tqdm.tqdm(rule_dict.items()):

        for section_label, section_text in rule_section.sections.items():
            emb = oaiapi.extract_embeddings(oaiapi.embeddings(section_text))[0]
            records.append(
                {
                    'rule_name': rule_name,
                    'section_label': section_label,
                    'text': section_text,
                    'emb': emb
                }
            )

    print('Saved embeddings to embeddings.pkl')
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(records, f)


if __name__ == '__main__':
    main()

