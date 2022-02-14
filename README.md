# LM-pretraining
## Efficient Pretraining of Language Models

### How to Run?
#### On a sample dataset
- Run `./run_sample.sh` (It downloads `wikitext-2-raw-v1`, trains tokenizer on it, and then trains BERT from scratch for 3 epochs)
#### On the Bookcorpus + English Wikipedia dataset
- Prepare the dataset by running `python3 prepare_bookcorpus_wiki.py`
- Train a tokenizer by running `./train_tokenizer.sh`
- Train BERT from scratch for 100,000 steps by running `./train_BERT.sh`

### Requirements

Run `pip install -r requirements.txt` to install dependencies
