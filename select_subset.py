####################################################################################
# This program does the following:
# 1. Load in the processed dataset(tokenized and chunked)
# 2. Read in the model weights from the checkpoint folder mentioned(to be taken as argument)
# 3. Compute the sentence embeddings of specified layer using these model weights
# 4. Perform subset selection, save the subset to the folder mentioned(to be taken as argument)
####################################################################################
import argparse
import math
import datetime
import time
import logging
import os
import sys
import datasets
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    BertConfig,
    BertTokenizerFast,
    BertModel,
    DataCollatorWithPadding,
)
from transformers.utils.versions import require_version
from cords.selectionstrategies.SL import SubmodStrategy
import pickle
from accelerate import InitProcessGroupKwargs
from helper_fns import taylor_softmax_v1
import numpy as np
import faiss

logger=get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

def parse_args():
    parser=argparse.ArgumentParser(description="Informative Subset Selection from text corpus using BERT embeddings as features")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="The directory to which the logs should be written"
    )
    parser.add_argument(
        "--subset_dir",
        type=str,
        required=True,
        help="The directory to which selected subsets should be written"
    )
    parser.add_argument(
        "--data_directory",
        type=str,
        default=None,
        help="The path to the directory containing the pre-processed huggingface dataset"
    )
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        required=True,
        help="Path to the pre-trained model checkpoint"
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=128,
        help="Batch size while computing model embeddings"
    )
    parser.add_argument(
        "--subset_fraction",
        type=float,
        default=0.25,
        help="Fraction of dataset to select for the subset"
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="fl",
        help="Subset Selection strategy"
    )
    parser.add_argument(
        "--partition_strategy",
        type=str,
        default="random",
        help="Partition strategy"
    )
    parser.add_argument(
        "--layer_for_embeddings",
        type=int,
        default=9,
        help="The hidden layer to use for the embeddings computation"
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=5000,
        help="Number of partitions for subset selection"
    )
    parser.add_argument(
        "--parallel_processes",
        type=int,
        default=96,
        help="Number of parallel processes for subset selection"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="LazyGreedy",
        help="Optimizer to use for submodular optimization"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature while calculating taylor softmax"
    )
    args=parser.parse_args()
    return args

def main():
    args=parse_args()
    init_process_group=InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=75000))
    accelerator=Accelerator(kwargs_handlers=[init_process_group])
    timestamp=datetime.datetime.now().strftime("%d_%m_%Y_%H.%M.%S")
    logging.basicConfig(
        filename=os.path.join(args.log_dir,f"subset_selection_{timestamp}.log"),
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info("Loading the dataset")
    dataset=load_from_disk(args.data_directory)

    logger.info(f"loading the model configuration.")
    config=BertConfig.from_pretrained("bert-base-uncased")
    
    logger.info(f"Loading the tokenizer.")
    tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")

    logger.info(f"Loading Model")
    model=BertModel(config)

    model.resize_token_embeddings(len(tokenizer))

    train_dataset=dataset["train"]
    train_dataset=train_dataset.remove_columns(["special_tokens_mask", "next_sentence_label"])

    num_samples=int(round(len(train_dataset)*args.subset_fraction, 0))

    logger.info(f"Full data has {len(train_dataset)} datapoints, subset data will have {num_samples} datapoints.")

    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)

    full_dataloader=DataLoader(
        train_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_batch_size
    )

    model, full_dataloader=accelerator.prepare(model, full_dataloader)

    if args.selection_strategy in ["fl", "logdet", "gc"]:
        subset_strategy=SubmodStrategy(logger, args.selection_strategy, 
                                        num_partitions=args.num_partitions, partition_strategy=args.partition_strategy,
                                        optimizer=args.optimizer, similarity_criterion="feature",
                                        metric="cosine", eta=1, stopIfZeroGain=False,
                                        stopIfNegativeGain=False, verbose=False, lambdaVal=1, sparse_rep=False)

    logger.info("Loading Checkpoint")
    accelerator.load_state(args.model_checkpoint_dir)
    pbar=tqdm(range(len(full_dataloader)), disable=not accelerator.is_local_main_process)
    model.eval()
    representations=[]
    batch_indices=[]
    total_cnt=0
    total_storage=0
    for step, batch in enumerate(full_dataloader):
        with torch.no_grad():
            output=model(**batch, output_hidden_states=True)
        embeddings=output["hidden_states"][args.layer_for_embeddings]
        mask=(batch["attention_mask"].unsqueeze(-1).expand(embeddings.size()).float())
        mask1=((batch["token_type_ids"].unsqueeze(-1).expand(embeddings.size()).float())==0)
        mask=mask*mask1
        mean_pooled=torch.sum(embeddings*mask, 1)/torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled=accelerator.gather(mean_pooled)
        total_cnt+=mean_pooled.size(0)
        if accelerator.is_main_process:
            mean_pooled=mean_pooled.cpu()
            total_storage+=sys.getsizeof(mean_pooled.storage())
            representations.append(mean_pooled)
        pbar.update(1)
    if accelerator.is_main_process:
        representations=torch.cat(representations, dim=0)
        representations=representations[:len(train_dataset)]
        total_storage+=sys.getsizeof(representations.storage())
        representations=representations.numpy()
        logger.info("Representations size: {}, Total number of samples: {}".format(total_storage/(1024*1024), total_cnt))
        batch_indices=list(range(len(train_dataset)))
        logger.info("Length of indices: {}".format(len(batch_indices)))
        logger.info("Representations gathered. Shape of representations: {}. Length of indices: {}".format(representations.shape, len(batch_indices)))
    if accelerator.is_main_process:
        partition_indices, greedyIdx, gains = subset_strategy.select(len(batch_indices)-1, batch_indices, representations, parallel_processes=args.parallel_processes, return_gains=True)
        probs=[]
        greedyList=[]
        gains=[]
        subset_indices=[[]]
        i=0
        for p in gains:
            greedyList.append(greedyIdx[i:i+len(p)])
            i+=len(p)
        probs=[taylor_softmax_v1(torch.from_numpy(np.array([partition_gains])/args.temperature)).numpy()[0] for partition_gains in gains]
        for i, partition_prob in enumerate(probs):
            rng=np.random.default_rng(int(time.time()))
            partition_budget=min(math.ceil((len(partition_prob)/len(batch_indices)) * num_samples), len(partition_prob)-1)
            subset_indices[0].extend(rng.choice(greedyList[i], size=partition_budget, replace=False, p=partition_prob).tolist())
        timestamp=os.path.basename(args.model_checkpoint_dir)
        output_file=f"partition_indices_{timestamp}.pkl"
        output_file=os.path.join(args.subset_dir, output_file)
        with open(output_file, "wb") as f:
            pickle.dump(partition_indices, f)
        output_file=f"subset_indices_{timestamp}.pt"
        output_file=os.path.join(args.subset_dir, output_file)
        torch.save(torch.tensor(subset_indices[0]), output_file)
        output_file=f"gains_{timestamp}.pkl"
        output_file=os.path.join(args.subset_dir, output_file)
        with open(output_file, "wb") as f:
            pickle.dump(gains, f)

if __name__=="__main__":
    main()