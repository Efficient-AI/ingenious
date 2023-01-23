import argparse
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from functools import reduce
import numpy as np
from tqdm.auto import tqdm
import time

def parse_args():
    parser=argparse.ArgumentParser(description="Evaluate GPT2 on CBT dataset")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2",
        help="path to gpt2 pretrained checkpoint"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device on which evaluation has to be run"
    )
    args=parser.parse_args()
    return args

def main():
    args=parse_args()

    tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token=tokenizer.eos_token
    model=GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model=model.to(f"cuda:{args.gpu}")
    model.eval()

    start_time=time.time()

    dataset=load_dataset("lambada")["test"]
    N=len(dataset)
    correct_cnt=0
    pbar=tqdm(range(N))
    for i in range(N):
        sentence=dataset[i]["text"]
        tokenized_sentence=tokenizer.tokenize(sentence)
        tensor_sentence=torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_sentence)])
        prompt_input_ids=tensor_sentence[:,:-1].to(model.device)
        outputs=model.generate(prompt_input_ids, do_sample=False, max_new_tokens=1)
        if outputs[0][-1]==tensor_sentence[0][-1]:
            correct_cnt+=1
        pbar.update(1)
    accuracy=(correct_cnt/N)*100.0
    print(accuracy)
    print(f"LAMBADA evaluation done on {args.model_name_or_path} in {time.time()-start_time} seconds")
    with open(f"{args.model_name_or_path}/lambada.txt", "w") as f:
        f.write(str(accuracy))
    
if __name__=="__main__":
    main()