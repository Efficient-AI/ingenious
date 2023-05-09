import argparse
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from functools import reduce
import numpy as np
from tqdm.auto import tqdm
import time

def parse_args():
    parser=argparse.ArgumentParser(description="Evaluate GPT2 on GRE Reading Comprehension dataset")
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

def get_predicted_choice(tokenizer, model, input_sentence, choices):
    losses=[]
    for i, choice in enumerate(choices):
        input_sentence=f"{input_sentence} {choice}"
        tokenize_input = tokenizer.tokenize(input_sentence)
        if len(tokenize_input)>1024:
            return None
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(model.device)
        loss = model(tensor_input, labels=tensor_input).loss
        losses.append(-loss.item())
    return choices[np.argmax(losses)]

def main():
    args=parse_args()

    tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
    model=GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model=model.to(f"cuda:{args.gpu}")
    model.eval()
    start_time=time.time()
    data=load_dataset("json", data_files="gre_reading_comprehension.json", field="examples")["train"]
    accuracy=0
    print(data)
    N=len(data)
    correct_preds=0
    ignored=0
    pbar=tqdm(range(N))
    for i in range(N):
        pred_choice=get_predicted_choice(tokenizer, model, data[i]["input"], list(data[i]["target_scores"].keys()))
        if pred_choice is None:
            ignored+=1
        elif data["target_scores"][pred_choice]==1:
            correct_preds+=1
        pbar.update(1)
    accuracy=(correct_preds)/(N-ignored)
    print(ignored, N)
    print(accuracy)
    print(f"GRE Reading Comprehension evaluation done on {args.model_name_or_path} in {time.time()-start_time} seconds")
    # with open(f"{args.model_name_or_path}/gre_reading_comprehension.txt", "w") as f:
    #     f.write(str(accuracy))

if __name__=="__main__":
    main()