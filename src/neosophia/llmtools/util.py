"""
Basic utilities and convenience functions.
"""

# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Ben Zimmer
# Cameron Fabbri
import re
import json
import pickle

from typing import Any, Dict, List, Optional, Tuple


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


def load_pickle(file_path: str) -> Any:
    """load a pickle file"""
    with open(file_path, 'rb') as file:
        res = pickle.load(file)
    return res


def save_pickle(obj: Any, file_path: str) -> None:
    """save an object to a pickle file"""
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def save_json(obj: Any, file_path: str) -> None:
    """
    save a data structure to a json file with nice
    default formatting
    """
    with open(file_path, 'w') as file:
        json.dump(
            obj=obj,
            fp=file,
            indent=2,
            sort_keys=False  # this preserves the order of dicts
        )


def read_lines(file_path: str) -> List[str]:
    """read a file into lines"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines
