import argparse
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from functools import reduce
import numpy as np
from tqdm.auto import tqdm
import time
import random
import evaluate

def parse_args():
    parser=argparse.ArgumentParser(description="Evaluate GPT2 on WMT-14 dataset")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2",
        help="path to gpt2 pretrained checkpoint"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=7,
        help="GPU device on which evaluation has to be run"
    )
    args=parser.parse_args()
    return args

def main():
    args=parse_args()

    tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
    model=GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model=model.to(f"cuda:{args.gpu}")
    model.eval()
    start_time=time.time()

    dataset=load_dataset("wmt14", "fr-en")
    N_train=len(dataset["train"])
    sample_size=7
        
    N=len(dataset["test"])
    
    language1="fr"
    language2="en"
    pbar=tqdm(range(N))
    predictions=[]
    references=[]

    fixed_context=[]
    for j in range(7):
        l1_sent=(dataset["train"][j]["translation"][language1])
        l2_sent=(dataset["train"][j]["translation"][language2])
        # tokenized_sent=tokenizer.tokenize(l1_sent+" = "+l2_sent+" . ")
        tokenized_sent=tokenizer.tokenize("FRENCH: "+l1_sent+"\nENGLISH: "+l2_sent+"\n")
        fixed_context.extend(tokenized_sent)
    for i in range(N):
        context=[]
        context.extend(fixed_context)
        l1_sent=(dataset["test"][i]["translation"][language1])
        # context.extend(tokenizer.tokenize(l1_sent+" = "))
        context.extend(tokenizer.tokenize("FRENCH: "+l1_sent+"\nENGLISH: "))
        tensor_input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(context)]).to(model.device)
        M=len(context)
        outputs=model.generate(tensor_input_ids, do_sample=False, max_length=30+M, pad_token_id=50256)
        predictions.append(tokenizer.decode(outputs[0][M:], skip_special_tokens=True).split("\n")[0])
        l2_sent=(dataset["test"][i]["translation"][language2])
        references.append(l2_sent)
        # if i>10:
        #     break
        pbar.update(1)
    bleu=evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)
    print(results)
    print(f"WMT-14 evaluation done on {args.model_name_or_path} in {time.time()-start_time} seconds")
    with open(f"{args.model_name_or_path}/wmt14-fr-en.txt", "w") as f:
        f.write(str(results))
    
if __name__=="__main__":
    main()