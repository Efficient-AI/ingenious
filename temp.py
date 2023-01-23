# import math
# import random
# import time
# import tqdm
# import submodlib
# import faiss
# import pickle
# import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances

# X=np.load("representations.npy")
# N=X.shape[0]

# representations=X[np.random.randint(0, N, size=10000)]

# dist_mat=euclidean_distances(representations)
# data_sijs=np.exp(-dist_mat/(0.1*dist_mat.mean()))

# obj = submodlib.FacilityLocationFunction(n = representations.shape[0], separate_rep=False, mode = 'dense', sijs = data_sijs)

# greedyList=obj.maximize(budget=representations.shape[0]-1, optimizer="LazyGreedy", stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, show_progress=True)

# gains=[p[1] for p in greedyList]

################################################

# from datasets import load_from_disk
# from transformers import BertTokenizerFast
# import torch
# import json
# import pandas as pd

# bert_first_sentences=load_from_disk("bert_first_sentences")
# bert_first_sentences=bert_first_sentences["train"]
# bert_first_sentences=bert_first_sentences.remove_columns(["token_type_ids", "attention_mask"])
# print("loaded")
# # bert_first_sentences.to_csv("bert_first_sentences.csv", num_proc=96)
# # bert_first_sentences.to_json("bert_first_sentences.json", num_proc=96, lines=False, orient="split")
# bert_first_sentences.to_json("bert_first_sentences.json", num_proc=96)

# from tqdm.auto import tqdm 
# import pickle

# def main():
#     freq_dict={}
#     with open("bert_first_sentences.json", "r") as f:
#         all_lines=f.readlines()
#     print("loaded bert_first_sentences")
#     pbar=tqdm(range(41543418))
#     for line in all_lines:
#         for id in line[14:-3].split(","):
#             if id in freq_dict:
#                 freq_dict[id]+=1
#             else:
#                 freq_dict[id]=1
#         pbar.update(1)
#     with open("fulldata-freqdict.pkl", "wb") as f:
#         pickle.dump(freq_dict, f)
    
# if __name__=="__main__":
#     main()

# from datasets import load_from_disk

# dataset=load_from_disk("wikitext-103-prepared")

# def add_indices(examples, indices):
#     examples["index"]=indices
#     return examples

# dataset=dataset.map(add_indices, with_indices=True, batched=True, num_proc=12)

# dataset.save_to_disk("wikitext-103-prepared-with-indices")

# import subprocess

# for i in [475000, 500000]:
#     subprocess.run(f"cp -r /home/hrenduchinta/LM-pretraining/models/bert_uncertainty_sampling_05_01_2023_13:21:30/step_{i} /home/hrenduchinta/LM-pretraining/models/bert_uncertainty_sampling_02_01_2023_08:33:38/step_{i}", shell=True)

# import pandas as pd
# import numpy as np
# import os
# import subprocess

# main_model_dir="/home/hrenduchinta/LM-pretraining/models/gpt2_ingenious_09_01_2023_10.22.38"
# for step in range(50000, 550000, 50000):
#     model_path=f"{main_model_dir}/step_{step}"
#     try:
#         data=pd.DataFrame(columns=["cola", "mrpc", "rte", "stsb", "sst2", "mnli-m", "mnli-mm", "qnli", "qqp"], index=[f"glue_run_{i}" for i in range(1, 21)])
#         for run in range(1, 21):
#             if run>5:
#                 tasks=["cola", "mrpc", "rte", "stsb"]
#             else:
#                 tasks=["cola", "mrpc", "rte", "stsb", "sst2", "mnli", "qnli", "qqp"]
#             l=[]
#             for task in tasks:
#                 with open(f"{model_path}/glue_run_{run}/{task}.log", "r") as f:
#                     lines=f.readlines()
#                     if task=="mnli":
#                         l.append(100*float(lines[-2].split()[-1].rstrip().rstrip('}')))
#                     l.append(100*float(lines[-1].split()[-1].rstrip().rstrip('}')))
#             if run>5:
#                 l.extend([np.nan]*5)
#             data.loc[f"glue_run_{run}"]=l
#         data.loc["average"]=data.mean(axis=0, skipna=True)
#         data.loc["stddev"]=data.std(axis=0, skipna=True)
#         data.loc["decrement_avg"]=[np.nan for i in range(9)]
#         data.loc["maximum"]=data.max(axis=0, skipna=True)
#         data.loc["decrement_max"]=[np.nan for i in range(9)]
#         data.to_csv(f"{model_path}/glue_results.csv")
#     except Exception as e:
#         print(f"step-{step}: {str(e)}")

# model_dir="/home/hrenduchinta/LM-pretraining/models/gpt2_ingenious_09_01_2023_10.22.38"
# os.makedirs(f"{model_dir}/results", exist_ok=True)
# for step in range(50000, 550000, 50000):
#     os.makedirs(f"{model_dir}/results/step_{step}", exist_ok=True)
#     subprocess.run(f"cp {model_dir}/step_{step}/glue_results.csv {model_dir}/results/step_{step}/glue_results.csv", shell=True)

# import subprocess

# main_model_dir="/home/hrenduchinta/LM-pretraining/models/gpt2_ingenious_09_01_2023_10:22:38"

# for step in range(50000, 550000, 50000):
#     for file in ["config.json", "merges.txt", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"]:
#         subprocess.run(
#             f"cp {main_model_dir}/{file} {main_model_dir}/step_{step}/{file}",
#             shell=True
#         )

# import os
# import subprocess

# main_model_dir="/home/hrenduchinta/LM-pretraining/models/gpt2_ingenious_09_01_2023_10.22.38"
# os.makedirs(f"{main_model_dir}/gpt2_ingenious_new", exist_ok=True)
# task="wmt14-fr-en"
# for step in range(50000, 550000, 50000):
#     subprocess.run(
#         f"cp {main_model_dir}/step_{step}/{task}.txt {main_model_dir}/gpt2_ingenious_new/{task}_{step}.txt",
#         shell=True
#     )
# import os
# import subprocess
# main_model_dir="/home/hrenduchinta/LM-pretraining/models/gpt2_random_fixed_13_01_2023_08.07.20"
# main_model_dirs=[
#     "/home/hrenduchinta/LM-pretraining/models/gpt2_vanilla_torch",
#     "/home/hrenduchinta/LM-pretraining/models/gpt2_ingenious_09_01_2023_10.22.38",
#     "/home/hrenduchinta/LM-pretraining/models/gpt2_random_fixed_13_01_2023_08.07.20",
#     "/home/hrenduchinta/LM-pretraining/models/gpt2_uncertainty_sampling_13_01_2023_22.17.06"
# ]
# for main_model_dir in main_model_dirs:
#     os.makedirs(f"{main_model_dir}/gpt2_random_fixed", exist_ok=True)
#     tasks=["webnlg_en", "common_gen"]
#     for task in tasks:
#         for step in range(50000, 550000, 50000):
#             subprocess.run(
#                 f"cp {main_model_dir}/step_{step}/{task}.txt {main_model_dir}/gpt2_random_fixed/{task}_{step}.txt",
#                 shell=True
#             )


# from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
# model=GPT2LMHeadModel.from_pretrained("gpt2")

# tokenizer.save_pretrained("models/gpt2")
# model.save_pretrained("models/gpt2")
# import os

# main_model_dir="/home/hrenduchinta/LM-pretraining/models/gpt2_random_fixed_13_01_2023_08.07.20"
# for step in range(50000, 550000, 50000):
#     for run in range(1, 21):
#         if run>5:
#             tasks=["cola", "mrpc", "rte", "stsb"]
#         else:
#             tasks=["cola", "mrpc", "rte", "stsb", "sst2", "mnli", "qnli"]
#         for task in tasks:
#             if not os.path.isfile(f"{main_model_dir}/step_{step}/glue_run_{run}/{task}.log"):
#                 print(step, run, task)

main_model_dirs=[
    "/home/hrenduchinta/LM-pretraining/models/gpt2_vanilla_torch",
    "/home/hrenduchinta/LM-pretraining/models/gpt2_ingenious_09_01_2023_10.22.38",
    "/home/hrenduchinta/LM-pretraining/models/gpt2_random_fixed_13_01_2023_08.07.20",
    "/home/hrenduchinta/LM-pretraining/models/gpt2_uncertainty_sampling_13_01_2023_22.17.06"
]

for main_model_dir in main_model_dirs:
    for step in range(50000, 550000, 50000):
        print(main_model_dir, step)
        with open(f"{main_model_dir}/step_{step}/openwebtext.log", "r") as f:
            for line in f:
                if "perplexity" in line:
                    print(line)