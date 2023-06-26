# neo-sophia
Applying the latest advancements in AI and machine learning to solve complex business problems.

## Data
[MSRB Regulatory Rulebook](https://www.msrb.org/sites/default/files/MSRB-Rule-Book-Current-Version.pdf)

To extract text from the MSRB Rulebook:

```
mkdir data
wget -P data https://www.msrb.org/sites/default/files/MSRB-Rule-Book-Current-Version.pdf
python -m scripts.extract_text
```
