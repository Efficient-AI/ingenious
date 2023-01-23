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
    parser=argparse.ArgumentParser(description="Evaluate GPT2 on Natural Questions dataset")
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

    dataset=load_dataset("nq_open")
    N_train=len(dataset["train"])
    sample_size=40
        
    N=len(dataset["validation"])

    pbar=tqdm(range(N))
    predictions=[]
    references=[]

    fixed_context=[]
    idxs=list(range(sample_size))
    for j in idxs:
        question=dataset["train"][j]["question"]
        answer=dataset["train"][j]["answer"][0]
        tokenized_sent=tokenizer.tokenize("QUESTION: "+question+"?\nANSWER: "+answer+"\n")
        fixed_context.extend(tokenized_sent)
    for i in range(N):
        context=[]
        context.extend(fixed_context)
        question=(dataset["validation"][i]["question"])
        context.extend(tokenizer.tokenize("QUESTION:"+question+"?\nANSWER: "))
        tensor_input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(context)]).to(model.device)
        M=len(context)
        outputs=model.generate(tensor_input_ids, do_sample=False, max_length=5+M, pad_token_id=50256)
        predictions.append(tokenizer.decode(outputs[0][M:], skip_special_tokens=True).split("\n")[0])
        answer=(dataset["validation"][i]["answer"])
        references.append(answer)
        # print(answer)
        # print(predictions[-1])

        # print("#"*80)
        # if i>100:
        #     break
        pbar.update(1)
    # bleu=evaluate.load("bleu")
    # results = bleu.compute(predictions=predictions, references=references)
    # print(results)
    # print(f"WMT-14 evaluation done on {args.model_name_or_path} in {time.time()-start_time} seconds")
    cnt=0
    for i in range(N):
        pred=predictions[i].strip().lower()
        for ref in references[i]:
            if ref.strip().lower()==pred:
                cnt+=1
    results={"accuracy":cnt/N}
    print(results)
    with open(f"{args.model_name_or_path}/nq.txt", "w") as f:
        f.write(str(results))
    
if __name__=="__main__":
    main()