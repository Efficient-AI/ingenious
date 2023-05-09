import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel
import evaluate
import torch
import time

def parse_args():
    parser=argparse.ArgumentParser(description="Evaluate GPT2 on Common Gen Dataset, GEM Benchmark")
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
    data=load_dataset("gem", "common_gen")

    def construct_input_for_batch(batch):
        """Construct input strings from a batch."""
        source = ["\nThis is a set of concepts: "+' '.join(concepts)+"\nThis is a sentence that uses the concepts: " for concepts in batch["concepts"]]
        target = batch["target"]
        return source, target

    def batch_tokenize(batch, tokenizer, max_length=32):
        """Construct the batch (source, target) and run them through a tokenizer."""
        source, target = construct_input_for_batch(batch)
        res = {
            "input_ids": tokenizer(source)["input_ids"],
            "labels": tokenizer(
                target,
                padding="max_length",
                truncation=True,
                max_length=max_length
            )["input_ids"],
        }
        return res

    MAX_LENGTH = 32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token=tokenizer.eos_token

    train_data_tokenized = data['train'].map(
        lambda batch: batch_tokenize(batch, tokenizer, max_length=MAX_LENGTH),
        batched=True
    )
    valid_data_tokenized = data['validation'].map(
        lambda batch: batch_tokenize(batch, tokenizer, max_length=MAX_LENGTH),
        batched=True
    )

    rouge_scorer = evaluate.load('rouge')

    DEVICE = f"cuda:{args.gpu}"
    BEAM_SIZE = 4

    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model = model.to(DEVICE)

    def beam_generate_sentences(
        batch,
        model,
        tokenizer,
        num_beams=4,
        max_length=32,
        device='cuda:0'
    ):
        """Generate outputs from a model with beam search decoding."""
        # Create batch inputs.
        source, _ = construct_input_for_batch(batch)
        # Use the model's tokenizer to create the batch input_ids.
        batch_features = tokenizer(source, padding=True, return_tensors='pt')
        # Move all inputs to the device.
        batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])

        # Generate with beam search.
        generated_ids = model.generate(
            **batch_features,
            num_beams=num_beams,
            max_length=max_length,
        )

        # Use model tokenizer to decode to text.
        generated_sentences = [
            tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
            for gen_ids in generated_ids
        ]
        return generated_sentences

    valid_output = data['validation'].map(
        lambda batch: {'generated': beam_generate_sentences(
            batch,
            model,
            tokenizer,
            num_beams=BEAM_SIZE,
            max_length=MAX_LENGTH,
            device=DEVICE)
        },
        batched=True,
        batch_size=128,
    )

    rouge_results = rouge_scorer.compute(
        predictions=valid_output["generated"],
        references=valid_output["target"],
    )

    for i in range(10):
        print("generated: ", valid_output["generated"][i])
        print("#"*80)
        print("target: ", valid_output["target"][i])
        print("#"*80)

    print(rouge_results)

    # with open(f"{args.model_name_or_path}/nq.txt", "w") as f:
    #     f.write(str(rouge_results))
    
if __name__=="__main__":
    main()