#!/bin/bash

pytest -s --tb=short \
    --cov-report term-missing --cov-report html:htmlcov/tests \
    --cov=src/neosophia/llmtools \
    src/neosophia

if [ "$(uname)" == "Darwin" ]; then
    open htmlcov/tests/index.html
fi
