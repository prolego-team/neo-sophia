"""
Simple project configuration.
"""

from typing import Any, Optional
import os
import json
import sys

CONFIG_FILE_PATH = 'config.json'

if os.path.isfile(CONFIG_FILE_PATH):
    with open(CONFIG_FILE_PATH, 'r') as config_file:
        CONFIG = json.load(config_file)
else:
    CONFIG = {}


def get_config(key: str, default: Optional[Any]) -> Any:
    """
    get a config parameter from the config file first,
    then from a provided default.
    """

    res = CONFIG.get(key)

    if res is not None:
        print(f'found config `{key}` in config file')
        return res

    res = default

    if res is not None:
        print(f'found config `{key}` from default value')
        return res

    print(f'No default value for config `{key}`.')
    print('Add it to the config.')
    sys.exit()


MODELS_DIR_PATH = get_config('MODELS_DIR_PATH', 'models')
DATASETS_DIR_PATH = get_config('DATASETS_DIR_PATH', 'data')
GENERATOR_CACHE_DIR_PATH = get_config('GENERATOR_CACHE_DIR_PATH', '.cache/generator')
OPENAI_API_KEY_FILE_PATH = get_config("OPENAI_API_KEY_FILE_PATH", 'openai_api_key.txt')
