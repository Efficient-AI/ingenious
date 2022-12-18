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
import subprocess

logger=get_logger(__name__)

def parse_args():
    parser=argparse.ArgumentParser(description="Informative Subset Selection from text corpus using BERT embeddings as features")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="The directory to which the logs should be written"
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default="fl",
        help="Subset Selection strategy"
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=5000,
        help="Number of partitions for subset selection"
    )
    parser.add_argument(
        "--partition_strategy",
        type=str,
        default="random",
        help="Partition strategy"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="LazyGreedy",
        help="Optimizer to use for submodular optimization"
    )
    parser.add_argument(
        "--subset_fraction",
        type=float,
        default=0.25,
        help="Fraction of dataset to select for the subset"
    )
    parser.add_argument(
        "--parallel_processes",
        type=int,
        default=96,
        help="Number of parallel processes for subset selection"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature while calculating taylor softmax"
    )
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        required=True,
        help="Path to the pre-trained model checkpoint"
    )
    parser.add_argument(
        "--subset_dir",
        type=str,
        required=True,
        help="The directory to which selected subsets should be written"
    )

    args=parser.parse_args()
    return args

def main():
    args=parse_args()
    init_process_group=InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=75000))
    accelerator=Accelerator(kwargs_handlers=[init_process_group])
    timestamp=datetime.datetime.now().strftime("%d_%m_%Y_%H.%M.%S")
    logging.basicConfig(
        filename=os.path.join(args.log_dir,f"submodular_optimization_{timestamp}.log"),
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
    
    if args.selection_strategy in ["fl", "logdet", "gc"]:
        subset_strategy=SubmodStrategy(logger, args.selection_strategy, 
                                        num_partitions=args.num_partitions, partition_strategy=args.partition_strategy,
                                        optimizer=args.optimizer, similarity_criterion="feature",
                                        metric="cosine", eta=1, stopIfZeroGain=False,
                                        stopIfNegativeGain=False, verbose=False, lambdaVal=1, sparse_rep=False)

    if accelerator.is_main_process:
        representations=np.load("representations.npy")
        batch_indices=list(range(representations.shape[0]))
        num_samples=int(round(representations.shape[0]*args.subset_fraction, 0))
        partition_indices, greedyIdx, gains = subset_strategy.select(len(batch_indices)-1, batch_indices, representations, parallel_processes=args.parallel_processes, return_gains=True)
        probs=[]
        greedyList=[]
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