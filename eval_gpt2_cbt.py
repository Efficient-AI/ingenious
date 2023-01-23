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

def get_predicted_choice(tokenizer, model, sentences, question, choices):
    context=reduce(lambda x, y: x+" "+y, sentences)
    question=question.replace("XXXXX", "{}")
    losses=[]
    for i, choice in enumerate(choices):
        input_sentence=context+question.format(choice)
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
    data_subsets=["CN", "NE", "P", "V"]
    accuracy={k:None for k in data_subsets}
    for subset in data_subsets:
        print(subset)
        data=load_dataset("cbt", subset)["test"]
        N=len(data)
        correct_preds=0
        ignored=0
        pbar=tqdm(range(N))
        for i in range(N):
            pred_choice=get_predicted_choice(tokenizer, model, data[i]["sentences"], data[i]["question"], data[i]["options"])
            if pred_choice is None:
                ignored+=1
            elif pred_choice==data[i]["answer"]:
                correct_preds+=1
            pbar.update(1)
        accuracy[subset]=(correct_preds)/(N-ignored)
        print(ignored, N)
        print(subset, accuracy[subset])
    print(accuracy)
    print(f"CBT evaluation done on {args.model_name_or_path} in {time.time()-start_time} seconds")
    with open(f"{args.model_name_or_path}/cbt.txt", "w") as f:
        f.write(str(accuracy))

if __name__=="__main__":
    main()