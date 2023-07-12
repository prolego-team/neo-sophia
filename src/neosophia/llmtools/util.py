"""
Basic utilities and convenience functions.
"""

import json
import pickle

from typing import Any, List


class Colors:
    """ANSI escape sequences for different colors."""
    RED = "\033[31m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    BLACK = "\033[30m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    WHITE = "\033[37m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"


def colorize(text: str, color: str) -> str:
    """apply a color to text"""
    return f"{color}{text}{Colors.RESET}"


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
