import subprocess
import datasets

def main():
    subprocess.run(["wget", "https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2"])
    subprocess.run(["tar", "-xf", "bookcorpus.tar.bz2"])
    bookcorpus1=datasets.Dataset.from_text("books_large_p1.txt")
    bookcorpus2=datasets.Dataset.from_text("books_large_p2.txt")
    bookcorpus=datasets.concatenate_datasets([bookcorpus1, bookcorpus2])
    wiki=datasets.load_dataset("wikipedia", "20200501.en", split="train")
    wiki=wiki.remove_columns(['title'])
    bert_dataset=datasets.concatenate_datasets([bookcorpus, wiki])
    bert_dataset=datasets.DatasetDict({"train": bert_dataset})
    bert_dataset.save_to_disk("./bert_dataset/")

if __name__=="__main__":
    main()