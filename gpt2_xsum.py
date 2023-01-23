import argparse
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from functools import reduce
import numpy as np
from tqdm.auto import tqdm
import time
import evaluate

def parse_args():
    parser=argparse.ArgumentParser(description="Evaluate GPT2 on XSum dataset")
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
    MAX_LENGTH=30
    NUM_BEAMS=4

    tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token=tokenizer.eos_token
    model=GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model=model.to(f"cuda:{args.gpu}")
    model.eval()

    start_time=time.time()

    dataset=load_dataset("json", data_files="xsum.json", field="examples")["train"]
    print(dataset)

    predictions=[]
    references=[]
    N=len(dataset)
    pbar=tqdm(range(N))
    for i in range(N):
        article=dataset[i]["input"]
        tokenized_article=tokenizer.tokenize("\nsummarize: "+article+"\none-sentence summary: ")
        M=len(tokenized_article)
        if M>1024-MAX_LENGTH:
            tokenized_article=tokenized_article[-(1024-MAX_LENGTH-1):]
        M=len(tokenized_article)
        input_ids=torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_article)]).to(model.device)
        outputs=model.generate(input_ids, num_beams=NUM_BEAMS, do_sample=True, max_length=MAX_LENGTH+M, pad_token_id=50256)
        predictions.append(tokenizer.decode(outputs[0][M:], skip_special_tokens=True).strip().split("\n")[0])
        references.append(dataset[i]["target"])
        # print("REFERENCE: ", references[-1])
        # print("PREDICTION: ", predictions[-1])
        # print("#"*80)
        # if i>10:
        #     break
        pbar.update(1)

    rouge=evaluate.load("rouge")
    results=rouge.compute(predictions=predictions, references=references)

    print(results)
    print(f"XSum Summarization done on {args.model_name_or_path} in {time.time()-start_time} seconds")
    with open(f"{args.model_name_or_path}/xsum.txt", "w") as f:
        f.write(str(results))

if __name__=="__main__":
    main()