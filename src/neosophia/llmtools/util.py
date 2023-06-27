"""
Basic utilities and convenience functions.
"""

# Copyright (c) 2023 Prolego Inc. All rights reserved.
# Ben Zimmer

from typing import Any, Dict, List, Optional, Tuple

import pickle
import json


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
