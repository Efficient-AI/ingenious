# LM-pretraining
## Efficient Pretraining of Language Models

### How to Run?

- First run `./train_BERT_tokenizer.sh` to train a tokenizer on the dataset
- Then run `./train_BERT_mlm.sh` to pre-train BERT on the dataset

**Note that all the default parameters correspond to the *bert-base-uncased* version of huggingface library**

#### To test the working of code on a sample dataset, set the following attribute(s) in both the shell scripts mentioned above
 - set the attribute `--dataset_name` to `wikitext`
 - set the attribute `--dataset_config_name` to `wikitext-2-raw-v1`

#### To start the full-fledged training on the Bookcorpus and Wikipedia combined, set the following attribute(s) in both the shell scripts mentioned above
- set the attribute `--dataset_name` to `kowndinya23/bert-corpus`
- REMOVE the attribute `--dataset_config_name`

### Note
- If running the BERT training on a TPU, it is better to pass the flag `--pad_to_max_length` in `run_BERT_pretraining_mlm.py` since dynamic padding slows down a TPU
- If running the BERT training on a GPU, it is better to **NOT** pass the flag `--pad_to_max_length` in `run_BERT_pretraining_mlm.py` since padding to max_length slows down a GPU

### Requirements

Run `pip install -r requirements.txt` to install dependencies
