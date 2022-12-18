from datasets import load_from_disk

dataset=load_from_disk("wikitext-103-prepared")

def remove_second_sent(examples):
    l=[]
    for i, sent in enumerate(examples["input_ids"]):
        idx=examples["token_type_ids"][i].index(1)
        examples["input_ids"][i][idx:]=[]
        examples["token_type_ids"][i][idx:]=[]
        examples["attention_mask"][i][idx:]=[]
        examples["special_tokens_mask"][i][idx:]=[]
    return examples

processed_dataset=dataset.map(remove_second_sent, batched=True, num_proc=96)

processed_dataset=processed_dataset.remove_columns(["special_tokens_mask", "next_sentence_label"])

processed_dataset.save_to_disk("wikitext_103-processed-first-sentences")