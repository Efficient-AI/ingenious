import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel
import evaluate
import torch
import time
from tqdm.auto import tqdm

def parse_args():
    parser=argparse.ArgumentParser(description="Evaluate GPT2 on E2E NLG Dataset")
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
    DEVICE = f"cuda:{args.gpu}"
    BEAM_SIZE = 4
    MAX_LENGTH = 32


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token=tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model = model.to(DEVICE)
    model.eval()

    dataset=load_dataset("json", data_files="e2e_nlg.json")["train"]
    print(dataset)

    predictions=[]
    references=[]

    start_time=time.time()
    N=len(dataset)
    pbar=tqdm(range(N))
    for i in range(N):
        context="\nThere is a restaurant with the following attributes: "+' '.join(dataset[i]["concepts"])+"\nThis is a sentence that uses the concepts: "
        tokenized_context=tokenizer.tokenize(context)
        M=len(tokenized_context)
        if M>1024-MAX_LENGTH:
            tokenized_context=tokenized_context[-(1024-MAX_LENGTH-1):]
        M=len(tokenized_context)
        input_ids=torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_context)]).to(model.device)
        outputs=model.generate(input_ids, max_length=MAX_LENGTH+M, pad_token_id=50256)
        predictions.append(tokenizer.decode(outputs[0][M:], skip_special_tokens=True).strip().split("\n")[0])
        references.append(dataset[i]["target"])
        pbar.update(1)

    rouge_scorer = evaluate.load('rouge')

    rouge_results = rouge_scorer.compute(
        predictions=predictions,
        references=references
    )

    print(rouge_results)
    print(f"Common Gen Dataset Evaluation done on {args.model_name_or_path} in {time.time()-start_time} seconds")
    with open(f"{args.model_name_or_path}/common_gen.txt", "w") as f:
        f.write(str(rouge_results))
    
if __name__=="__main__":
    main()