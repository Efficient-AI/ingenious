import argparse
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from functools import reduce
import numpy as np
from tqdm.auto import tqdm
import time

def parse_args():
    parser=argparse.ArgumentParser(description="Evaluate GPT2 on Strategy QA dataset")
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

def get_predicted_choice(tokenizer, model, input_sentence):
    losses=[]
    for choice in ["Yes", "No"]:
        input_sentence=f"{input_sentence} {choice}."
        tokenize_input = tokenizer.tokenize(input_sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(model.device)
        loss = model(tensor_input, labels=tensor_input).loss
        losses.append(-loss.item())
    print(type(losses[0]))
    if losses[0]>losses[1]:
        return "Yes"
    else:
        return "No"

def main():
    args=parse_args()

    tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
    model=GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model=model.to(f"cuda:{args.gpu}")
    model.eval()
    start_time=time.time()

    data=load_dataset("json", data_files="strategy_qa.json", field="examples")["train"]
    accuracy=0
    N=len(data)
    correct_preds=0
    pbar=tqdm(range(N))
    for i in range(N):
        pred_choice=get_predicted_choice(tokenizer, model, data[i]["input"])
        if data[i]["target_scores"][pred_choice]==1:
            correct_preds+=1
        print(data[i]['input'])
        print(data[i]["target_scores"])
        print(pred_choice)
        print("#"*80)
        if i>10:
            break
        pbar.update(1)
    accuracy=correct_preds/N
    print(accuracy)
    print(f"Strategy QA evaluation done on {args.model_name_or_path} in {time.time()-start_time} seconds")
    with open(f"{args.model_name_or_path}/strategy_qa.txt", "w") as f:
        f.write(str(accuracy))

if __name__=="__main__":
    main()