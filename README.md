# Project: Learning to write notes in EHR with LLaMA

## Usage

- Install Python 3.11
- Install Python Dependency Manager (https://pdm-project.org/en/latest/)
- Type ```pdm install``` in repository root directory

- Put MIMIC-III csv files into data/mimic-iii-clinical-database-1.4/
- Run main.py to process and filter the data
- Run train.py to train the model
- Run evaluation.py to get the outputs for the test dataset
- Run get_metrics.py to print the ROUGE scores