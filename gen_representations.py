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
    dataloader=DataLoader(dataset, collate_fn=data_collator, batch_size=256)

    model, dataloader=accelerator.prepare(model, dataloader)

    model.eval()
    representations=[]
    progressbar=tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs=model(**batch, output_hidden_states=True)
        embeddings=outputs["hidden_states"][1]
        mask=(batch['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float())
        mask1=((batch['token_type_ids'].unsqueeze(-1).expand(embeddings.size()).float())==0)
        mask=mask*mask1
        mean_pooled=torch.sum(embeddings*mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled=accelerator.gather(mean_pooled)
        if accelerator.is_main_process:
            mean_pooled=mean_pooled.cpu()
            representations.append(mean_pooled)
        progressbar.update(1)
    
    if accelerator.is_main_process:
        representations=torch.cat(representations, dim=0)
        torch.save(representations, "./bert_representations_layer_1.pt")

if __name__=="__main__":
    main()