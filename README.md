# LLM Strategy from Prolego

Large-scale language models like GPT-4 from OpenAI serve as foundational technologies that can be applied to virtually any business issue. However, the robust power and flexibility of this technology come with a significant challenge: it is extremely difficult to pinpoint the optimal opportunities for leveraging this technology within a company. 

This project is designed to assist analytics leaders, product managers, and development teams in surmounting these obstacles by demonstrating the technology's application across a variety of common business problems. The project unfolds through a series of episodes, each accompanied by the following resources:

1. Walkthrough videos available on [Prolego's YouTube Channel](https://www.youtube.com/@prolego9489).
2. Tagged releases on the main branch of this repository.
3. Conversations held within [Prolego's Discord community](#).

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

### Packaging and Distributing

This project uses a [`pyproject.toml` file for packaging](https://packaging.python.org/en/latest/tutorials/packaging-projects/).  This file contains project metadata and a list of requirements for the core library.

From the project root run:

    python -m build
    python3 -m twine upload --repository testpypi dist/*

# Episodes
## Episode 1 - Embeddings, Semantic search and Document Q&A
Most companies are struggling to pick the best AI use cases from many different options. By building a core competency in document embeddings you can begin developing a set of capabilities applicable for many enterprise use cases. In Episoide 1 we provide a primer on embeddings for a business audience and demonstrate the use of embeddings in semantic search and document Q&A. 

Videos
- [Document embeddings are foundational capabilities for your AI strategy](#)
- [Document embeddings - technical walkthrough](#)

### Running the Episode 1 application
First checkout Episode 1, [Release v0.1.0](https://github.com/prolego-team/neo-sophia/releases/tag/v0.1.0)
````
git checkout tags/v0.1.0
````

Questions? Just ask in our [Discord Community](#).

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


# About Prolego & Neo Sophia

[Prolego](https://prolego.com) is an AI services company that started in 2017 and has helped some of the world’s biggest companies generate opportunities with AI. "Prolego" is the Greek word for "predict". We needed a name for this repo and decided to use the Greek words for "new" (neo) and "wisdom" (sophia). And we just thought that Neo Sophia sounded cool.

The team:

* [Kevin Dewalt](https://github.com/kevindewalt)
* [Justin Pounders](https://github.com/jmpounders)
* [Cameron Fabbri](https://github.com/cameronfabbri)
* [Ben Zimmer](https://github.com/bdzimmer)
