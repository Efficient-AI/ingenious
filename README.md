# LM-pretraining
## Efficient Pretraining of Language Models
### Environment Setup
#### Run the following commands in a sequence
- `conda create -n ingenious -c rapidsai -c nvidia -c conda-forge cuml=22.02 python=3.8 cudatoolkit=11.4`
- `conda activate ingenious`
- `pip3 install -r requirements.txt`
- `pip3 install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib
`
#### Configuring the accelerate library according to the training environment
Run `accelerate config` and answer the following questions
An example is given below
- In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): **0**
- Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU): **2**
- How many different machines will you use (use more than 1 for multi-node training)? [1]: **1**
- Do you want to use DeepSpeed? [yes/NO]: **NO**
- How many processes in total will you use? [1]: **4**
- Do you wish to use FP16 (mixed precision)? [yes/NO]: **yes**

##### Change the required parameters in `train_sample.py` or `train_BERT.py` 
### On a sample dataset
- Run `python3 prepare_sample.py` (It downloads `wikitext-103-raw-v1`)
- Run `python3 train_sample.py` trains tokenizer on data, and then trains BERT from scratch for 3 epochs)
### On the Bookcorpus + English Wikipedia dataset
- Download the `bert_dataset_prepared` folder from <a href="https://drive.google.com/drive/folders/1pqgZLnEQJjf7v4OyQ2IjbnjBtk_Kqgrf?usp=sharing">here</a>.
- Train BERT from scratch for 1,000,000 steps by running `python3 train_BERT.py`