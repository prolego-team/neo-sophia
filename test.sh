#!/bin/bash

pytest -s --tb=short \
    --cov-report term-missing --cov-report html:htmlcov/tests \
    --cov=src/neosophia/llmtools \
    --cov=src/neosophia/db \
    --cov=src/neosophia/datasets \
    --cov=src/neosophia/agents \
    --cov=src/neosophia/search \
    --cov=examples \
    src/neosophia \
    examples

if [ "$(uname)" == "Darwin" ]; then
    open htmlcov/tests/index.html
fi
