import os
import subprocess
from datetime import datetime
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="test subset script")
    parser.add_argument(
        "--visible_gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="visible_gpus"
    )
    parser.add_argument(
        "--main_process_port",
        type=int,
        default=55555,
        help="main process port for accelerate launch"
    )
    args=parser.parse_args()
    return args

def main():
    args=parse_args()
    now=datetime.now()
    timestamp=now.strftime("%d_%m_%Y_%H.%M.%S")
    log_dir=f"./logs/test_logs_{timestamp}/"
    subset_dir=f"./subsets/test_subsets_{timestamp}/"
    partitions_dir=f"./partitions/test_partitions_{timestamp}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(subset_dir, exist_ok=True)
    os.makedirs(partitions_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_gpus
    l=[
        "accelerate", "launch", "--main_process_port", f"{args.main_process_port}", "select_subset.py",
        "--log_dir", log_dir,
        "--subset_dir", subset_dir,
        # "--partitions_dir", partitions_dir,
        "--data_directory", "bert_first_sentences",
        "--model_checkpoint_dir", "/home/hrenduchinta/LM-pretraining/models/fl_bert_23_11_2022_06:34:28/step_250000",
        "--per_device_batch_size", "128",
        "--subset_fraction", "0.25",
        "--selection_strategy", "fl",
        "--partition_strategy", "kmeans_clustering",
        "--layer_for_embeddings", "9",
        "--num_partitions", "5000",
        "--parallel_processes", "96",
        "--optimizer", "LazyGreedy",
        "--temperature", "1.0",
    ]
    subprocess.run(l)

if __name__=="__main__":
    main()