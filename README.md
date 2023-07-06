# neo-sophia

Applying the latest advancements in AI and machine learning to solve complex business problems.

#### Environment Setup

Create Anaconda environment:

    conda env create -f env.yml
    conda activate neosophia

Editable install of `neosophia` package. 

    cd ..
    pip install -e neo-sophia
    cd neo-sophia

Note that installation is required to run tests and examples due due "src layout"!

Run tests:

    ./test.sh

## Demos and Examples

- `examples/chromadb_example.py`:  This script creates a Chroma vector database and runs a natural language query against the database.