import os
import subprocess
from datetime import datetime
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="train BERT")
    parser.add_argument(
        "--visible_gpus",
        type=str,
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
    timestamp=now.strftime("%d_%m_%Y_%H:%M:%S")
    log_dir="./logs/bert_uncertainty_sampling_"+timestamp+"/"
    model_dir="./models/bert_uncertainty_sampling_"+timestamp +"/"
    subset_dir="./subsets/bert_uncertainty_sampling_"+timestamp+"/"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_gpus
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(subset_dir, exist_ok=True)
    l=[
        "accelerate", "launch", "--main_process_port", f"{args.main_process_port}", "run_mlm_nsp_uncertainty_sampling.py",
        "--preprocessed",
        "--log_dir", log_dir,
        "--subset_dir", subset_dir,
        "--load_data_from_disk",
        "--data_directory", "./bert_dataset_prepared_with_indices",
        "--tokenizer_name", "bert-base-uncased",
        # "--model_type", "bert",
        "--preprocess_batch_size", "2000",
        "--per_device_train_batch_size", "128",
        "--per_device_eval_batch_size", "128",
        "--learning_rate", "5.5555656565656565e-05",
        "--lr_max_steps", "550000",
        "--weight_decay" ,"0.01",
        "--max_train_steps", "50000",
        "--gradient_accumulation_steps", "1",
        "--num_warmup_steps", "0",
        "--output_dir", model_dir,
        "--model_type", "bert",
        "--max_seq_length", "128",
        "--preprocessing_num_workers", "96",
        "--mlm_probability" ,"0.15",
        "--short_seq_prob", "0.1",
        "--nsp_probability", "0.5",
        "--subset_fraction", "0.25",
        "--checkpointing_steps", "25000",
        "--resume_from_checkpoint", "/home/hrenduchinta/LM-pretraining/models/bert_uncertainty_sampling_02_01_2023_08:33:38/step_450000"
    ]
    # with open(f"{log_dir}parameters.txt", "w") as f:
    #     for i in l:
    #         f.write(f"{i}\n")
    subprocess.run(l)

if __name__=="__main__":
    main()