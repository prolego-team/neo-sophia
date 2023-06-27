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

Run tests:

    ./test.sh

#### Data
[MSRB Regulatory Rulebook](https://www.msrb.org/sites/default/files/MSRB-Rule-Book-Current-Version.pdf)

To extract text from the MSRB Rulebook:

```
mkdir data
wget -P data https://www.msrb.org/sites/default/files/MSRB-Rule-Book-Current-Version.pdf
python -m scripts.extract_text
```
Note that installation is required to run tests and examples due due "src layout"!

Run tests:

    ./test.sh
