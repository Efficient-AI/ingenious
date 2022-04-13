import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import BertModel, BertTokenizerFast, DataCollatorWithPadding

def main():
    accelerator=Accelerator()
    dataset=load_from_disk("./bert_dataset_prepared")
    tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")
    model=BertModel.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))
    dataset=dataset["train"]
    dataset=dataset.remove_columns(["special_tokens_mask","next_sentence_label"])
    dataset.set_format("torch")
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader=DataLoader(dataset, collate_fn=data_collator, batch_size=32)

    model, dataloader=accelerator.prepare(model, dataloader)

    model.eval()
    all_outputs=[]
    progressbar=tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs=model(**batch)
        #hidden_states=torch.moveaxis(torch.stack(outputs.hidden_states),1,0)
        embeddings=outputs.last_hidden_state
        mask=(batch['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float())
        mean_pooled=torch.sum(embeddings*mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        all_outputs.append(accelerator.gather(mean_pooled).to("cpu"))
        progressbar.update(1)

    accelerator.save(torch.cat(all_outputs), "./bert_dataset_representations.pt")

if __name__=="__main__":
    main()