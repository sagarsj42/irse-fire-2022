# IRSE FIRE 2022

Information Retrieval in Software Engineering, FIRE '22

Relevance of a given comment to code snippet.

### Team: iREL, IIIT-H

## Prerequisites
- transformers
- datasets
- scikit-learn
- numpy
- pandas

## Notebooks

- data_exploration.ipynb: EDA on the data
- prepare_data.ipynb: Preparing de-duplicated data files to be used for training. For test data, the samples overlapping with train are removed as well
- train_[bert|roberta|codebert].ipynb - Training & evaluation the task using HuggingFace's Trainer API. Changing the MODEL_KEY and EXP_NAME to apropriate values is sufficient to perform different experiments
- prepare_output.ipynb - Preparing the primary results output file
