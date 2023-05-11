# INGENIOUS
## Data Efficient Pre-training of Large Language Models
### Environment Setup
#### Run the following in a sequence to set up the environment for running the code. (It is assumed that you have anaconda installed)
>- `conda create --name ingenious python=3.9`
>- `conda activate ingenious`
>- `conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch`
>- `pip3 install -r requirements.txt`
>- `git clone https://github.com/Efficient-AI/submodlib.git`
>- `cd submodlib`
>- `pip3 install .`
>- `conda install -c conda-forge faiss-gpu`
>- `cd ..`
>- `conda deactivate`
#### Configuring the accelerate library according to the training environment
Run `accelerate config` and answer the questions that follow.
An example is given below
- In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): **0**
- Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU): **2**
- How many different machines will you use (use more than 1 for multi-node training)? [1]: **1**
- Do you want to use DeepSpeed? [yes/NO]: **NO**
- How many processes in total will you use? [1]: **8**
- Do you wish to use FP16 (mixed precision)? [yes/NO]: **NO**

### Running the Code
Change appropriate parameters in `train_mlm_nsp.py` and run it as - `python3 train_mlm_nsp.py`  to train the model using INGENIOUS.