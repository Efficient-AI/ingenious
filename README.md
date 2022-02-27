# LM-pretraining
## Efficient Pretraining of Language Models

### How to Run?
#### Configuring the accelerate library according to the trianing environment
Run `accelerate config` and answer the following questions
An example is given below
- In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): **0**
- Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU): **2**
- How many different machines will you use (use more than 1 for multi-node training)? [1]: **1**
- Do you want to use DeepSpeed? [yes/NO]: **NO**
- How many processes in total will you use? [1]: **8**
- Do you wish to use FP16 (mixed precision)? [yes/NO]: **yes**

#### On a sample dataset
- Run `./run_sample.sh` (It downloads `wikitext-2-raw-v1`, trains tokenizer on it, and then trains BERT from scratch for 3 epochs)
#### On the Bookcorpus + English Wikipedia dataset
- Prepare the dataset by running `python3 prepare_bookcorpus_wiki.py`
- Train a tokenizer by running `./train_tokenizer.sh`
- Train BERT from scratch for 100,000 steps by running `./train_BERT.sh`

### Requirements

Run `pip install -r requirements.txt` to install dependencies
