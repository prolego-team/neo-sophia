# NeoSophia

Applying the latest advancements in AI and machine learning to solve complex business problems.

## Development Environment Setup

### Anaconda Environment Setup

Create Anaconda environment:

    conda env create -f env.yml
    conda activate neosophia

### Installation From Source

Editable install of `neosophia` package.

    cd ..
    pip install -e neo-sophia
    cd neo-sophia

Note that installation is required to run tests and examples due due "src layout"!

### Running Tests

Run tests:

    ./test.sh

## Getting Data

[MSRB Regulatory Rulebook](https://www.msrb.org/sites/default/files/MSRB-Rule-Book-Current-Version.pdf)

To extract text from the MSRB Rulebook:

```
python -m scripts.download_and_extract_msrb
```

## Running The MSRB Semantic Search Example

First create a configuration file named `config.json` in the project root directory with the following contents.  

```
{
    "MODELS_DIR_PATH": "models",
    "DATASETS_DIR_PATH": "data",
    "GENERATOR_CACHE_DIR_PATH": "cache",
    "OPENAI_API_KEY_FILE_PATH": "openai_key_file"
}
```

The paths can be whatever you like.  The OpenAI key file should be a text file containing your OpenAI API key.

Now you can start the semantic search demo by running

    python -m examples.interface

## Packaging and Distributing

This project uses a [`pyproject.toml` file for packaging](https://packaging.python.org/en/latest/tutorials/packaging-projects/).  This file contains project metadata and a list of requirements for the core library.

From the project root run:

    python -m build
    python3 -m twine upload --repository testpypi dist/*