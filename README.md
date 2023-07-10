# LLM Strategy from Prolego

Large-language models like GPT-4 from OpenAI are foundational technologies that can be applied to almost any business problem. Unfortunately the power and flexibility of this technology comes at a cost: it is extremely difficult to identify the best opportunities for leveraging the technology within any company. 

This project is designed to help analytics leaders, product managers, and development teams overcome these challenges by seeing the technology applied to a variety of common business problems. The project is broken into a series of Episodes, each of which has the following resources.

1. Walkthrough videos on [Prolego's YouTube Channel](https://www.youtube.com/@prolego9489).
2. Tagged releases on the main branch of this repo.
3. Conversations in [Prolego's Discord]()

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

# Episodes
## Episode 1 - Embeddings, Semantic search and Document Q&A
Most companies are struggling to pick the best AI use cases from many different options. By building a core competency in document embeddings you can begin developing a set of capabilities applicable for many enterprise use cases. In Episoide 1 we provide a primer on embeddings for a business audience and demonstrate the use of embeddings in semantic search and document Q&A. 

Videos
[Document embeddings are foundational capabilities for your AI strategy](#)
[Document embeddings - technical walkthrough](#)

### Running the Episode 1 application

This episode uses data from the [MSRB Regulatory Rulebook](https://www.msrb.org/sites/default/files/MSRB-Rule-Book-Current-Version.pdf)

1. Extract text from the MSRB Rulebook:

```
python -m scripts.download_and_extract_msrb
```

2. Create a configuration file named `config.json` in the project root directory with the following contents.  

```
{
    "MODELS_DIR_PATH": "models",
    "DATASETS_DIR_PATH": "data",
    "GENERATOR_CACHE_DIR_PATH": "cache",
    "OPENAI_API_KEY_FILE_PATH": "openai_key_file"
}
```

The paths can be whatever you like.  The OpenAI key file should be a text file containing your OpenAI API key.

3. Start the semantic search demo by running

    python -m examples.interface

## Packaging and Distributing

This project uses a [`pyproject.toml` file for packaging](https://packaging.python.org/en/latest/tutorials/packaging-projects/).  This file contains project metadata and a list of requirements for the core library.

From the project root run:

    python -m build
    python3 -m twine upload --repository testpypi dist/*
