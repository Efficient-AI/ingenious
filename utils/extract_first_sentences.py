from transformers import BertTokenizerFast
from datasets import load_from_disk

dataset=load_from_disk('bert_dataset_prepared')
dataset=dataset["train"]
tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")

def extract_first_sentences(examples):
    for i, input_ids in enumerate(examples["input_ids"]):
        idx=input_ids.index(tokenizer.sep_token_id)
        examples["input_ids"][i]=input_ids[:idx+1]
        examples["attention_mask"][i]=examples["attention_mask"][i][:idx+1]
        examples["token_type_ids"][i]=examples["token_type_ids"][i][:idx+1]
        examples["special_tokens_mask"][i]=examples["special_tokens_mask"][i][:idx+1]
    return examples

# filter points from dataset with next_sentence_label == 0
nsp_zero=dataset.filter(lambda examples: [x==0 for x in examples["next_sentence_label"]], batched=True, num_proc=96, keep_in_memory=True)
nsp_zero.save_to_disk("nsp_zero")
# filter points from dataset with next_sentence_label == 1
nsp_one=dataset.filter(lambda examples: [x==1 for x in examples["next_sentence_label"]], batched=True, num_proc=96, keep_in_memory=True)
nsp_one.save_to_disk("nsp_one")
# extract first sentences from both datasets
nsp_zero=nsp_zero.map(extract_first_sentences, batched=True, num_proc=96, remove_columns=["next_sentence_label", "special_tokens_mask"], keep_in_memory=True)
nsp_one=nsp_one.map(extract_first_sentences, batched=True, num_proc=96, remove_columns=["next_sentence_label", "special_tokens_mask"], keep_in_memory=True)

# save datasets
nsp_zero.save_to_disk("first_sent_nsp_zero")
nsp_one.save_to_disk("first_sent_nsp_one")