import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import BertForPreTraining, BertTokenizerFast, DataCollatorForLanguageModeling
import time
from torch.nn import CrossEntropyLoss

def main():
    accelerator=Accelerator()
    dataset=load_from_disk("./bert_dataset_prepared_with_indices")
    tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")
    model=BertForPreTraining.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))
    dataset=dataset["train"]
    # dataset=dataset.remove_columns(["special_tokens_mask","next_sentence_label"])
    dataset.set_format("torch")
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer)
    dataloader=DataLoader(dataset, collate_fn=data_collator, batch_size=128)
    vocab_size=model.config.vocab_size
    model, dataloader=accelerator.prepare(model, dataloader)

    model.eval()
    losses=[]
    indices=[]
    loss_fct = CrossEntropyLoss(reduction='none')
    progressbar=tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
    for step, batch in enumerate(dataloader):
        idx=batch.pop("index", None)
        with torch.no_grad():
            outputs=model(**batch)
        mask=batch["labels"]!=(-100)
        masked_lm_loss = loss_fct(torch.transpose(outputs.prediction_logits,1,2), batch["labels"])
        masked_lm_loss = masked_lm_loss.sum(dim=1)/mask.sum(dim=1)
        next_sentence_loss = loss_fct(outputs.seq_relationship_logits.view(-1, 2), batch["next_sentence_label"].view(-1))
        loss=masked_lm_loss+next_sentence_loss
        loss=accelerator.gather(loss)
        if accelerator.is_main_process:
            losses.append(loss)
            indices.append(idx)
        progressbar.update(1)
    
    if accelerator.is_main_process:
        losses=torch.cat(losses, dim=0)
        losses=losses[:len(dataset)]
        indices=torch.cat(indices, dim=0)
        indices=indices[:len(dataset)]
        torch.save(losses, "./losses.pt")

if __name__=="__main__":
    main()