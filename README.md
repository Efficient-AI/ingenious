# LM-pretraining
## Efficient Pretraining of Language Models

### How to Run?

- Edit the dataset name in the train_BERT_tokenizer.sh along with any optional changes in parameters mentioned in it. Then run `./train_BERT_tokenizer.sh` to train a tokenizer on the dataset
- Edit the dataset name in the train_BERT_mlm.sh along with any optional changes in parameters mentioned in it. Then run `./train_BERT_mlm.sh` to pre-train BERT on the dataset

### Requirements

Run `pip install -r requirements.txt` to install dependencies