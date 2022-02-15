import datasets
from datasets import load_dataset

dataset=load_dataset("wikitext", "wikitext-2-raw-v1", split='train')
dataset=datasets.DatasetDict({"train": dataset})
dataset.save_to_disk("./wikitext-2-raw-v1")