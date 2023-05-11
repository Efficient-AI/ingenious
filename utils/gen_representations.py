import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import BertModel, BertForPreTraining, BertTokenizerFast, DataCollatorWithPadding, DataCollatorForLanguageModeling
import time

def main():
    accelerator=Accelerator()
    dataset=load_from_disk("./bert_dataset_prepared_10000")
    tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")
    # model=BertModel.from_pretrained("bert-base-uncased")
    model=BertForPreTraining.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))
    # dataset=dataset.remove_columns(["special_tokens_mask", "next_sentence_label"])
    dataset.set_format("torch")
    # data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    dataloader=DataLoader(dataset, collate_fn=data_collator, batch_size=16)

    model, dataloader=accelerator.prepare(model, dataloader)

    model.eval()
    representations=[]
    progressbar=tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs=model(**batch, output_hidden_states=True)
        embeddings=outputs["hidden_states"][9]
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
        representations=representations[:len(dataset)]
        torch.save(representations, "./bert_sample_representations_1.pt")

    # model.eval()
    # losses=[]
    # progressbar=tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
    # for step, batch in enumerate(dataloader):
    #     with torch.no_grad():
    #         outputs=model(**batch)
    #     loss=outputs.loss
    #     loss=accelerator.gather(loss)
    #     if accelerator.is_main_process:
    #         losses.append(loss)
    #     progressbar.update(1)
    
    # if accelerator.is_main_process:
    #     losses=torch.cat(losses, dim=0)
    #     torch.save(losses, "./losses.pt")

if __name__=="__main__":
    main()