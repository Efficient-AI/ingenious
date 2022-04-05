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
- How many processes in total will you use? [1]: **4**
- Do you wish to use FP16 (mixed precision)? [yes/NO]: **yes**

#### On a sample dataset
- Run `python3 prepare_sample.py` (It downloads `wikitext-2-raw-v1`)
- Run `python3 train_sample.py` trains tokenizer on data, and then trains BERT from scratch for 3 epochs)
#### On the Bookcorpus + English Wikipedia dataset
- Prepare the dataset by running `python3 prepare_bookcorpus_wiki.py`
- Train BERT from scratch for 1,000,000 steps by running `python3 train_BERT.py`

### Requirements

Run `pip install -r requirements.txt` to install dependencies

# [GLUE Benchmark](https://gluebenchmark.com/) 

The General Language Understanding Evaluation(GLUE) benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems. GLUE consists of:
- A benchmark of nine sentence- or sentence-pair language understanding tasks built on established existing datasets and selected to cover a diverse range of dataset sizes, text genres, and degrees of difficulty,
- A diagnostic dataset designed to evaluate and analyze model performance with respect to a wide range of linguistic phenomena found in natural language, and
- A public leaderboard for tracking performance on the benchmark and a dashboard for visualizing the performance of models on the diagnostic set.

## GLUE Tasks
| Name | CODE| Metric | `bert-base-uncased`|
|----------|---- |--------|-----|
|The Corpus of Linguistic Acceptability|CoLA| Matthew's Corr|49.23|
|The Stanford Sentiment Treebank|SST-2|Accuracy|91.97|
|Microsoft Research Paraphrase Corpus|MRPC|F1/Accuracy|89.47/85.29|
|Semantic Textual Similarity Benchmark|STS-B|Pearson-Spearman Corr|	83.95/83.70|
|Quora Question Pairs|QQP|F1/Accuracy|88.40/84.31|
|MultiNLI Matched|MNLI|Accuracy|80.61|
|MultiNLI Mismatched|MNLI|Accuracy|81.08|
|Question NLI|QNLI|Accuracy|87.46|
|Recognizing Textual Entailment|RTE|Accuracy|61.73|
|Winograd NLI|WNLI|Accuracy|45.07|
|Diagnostics Main|diagnostic|Matthew's Corr|