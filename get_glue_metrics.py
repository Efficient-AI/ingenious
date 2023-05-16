import os
import subprocess
import argparse
import time

def parse_args():
    parser=argparse.ArgumentParser(description="GLUE")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to pytorch saved model"
    )
    parser.add_argument(
        "--main_process_port",
        type=str,
        default=55555,
        help="main process port for huggingface accelerate"
    )
    parser.add_argument(
        "--visible_gpus",
        type=str,
        required=True,
        help="gpu number to use"
    )
    args=parser.parse_args()
    return args

def main():
    args=parse_args()
    model_name_or_path=args.model_dir
    tasks=["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "mnli", "qqp"]
    for run in range(1, 21):
        os.makedirs(os.path.join(model_name_or_path, f"glue_run_{run}"), exist_ok=True)
    for i, task in enumerate(tasks):
        if task in ["cola", "mrpc", "rte", "stsb"]:
            num_runs=20
        else:
            num_runs=5
        for run in range(1, num_runs+1):
            os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_gpus
            l=[
                "accelerate", "launch", "--main_process_port", f"{args.main_process_port}", "run_glue.py",
                "--log_file", os.path.join(model_name_or_path, f"glue_run_{run}", f"{task}.log"),
                "--task_name", task,
                "--max_length", "128",
                "--model_name_or_path", model_name_or_path,
                "--per_device_train_batch_size", "32",
                "--per_device_eval_batch_size", "32",
                "--learning_rate", f"5e-5",
                "--weight_decay" ,"0.0",
                "--num_train_epochs", "3",
                "--seed", f"{run}",
            ]
            subprocess.run(l)
            # wait for 1 minute
            time.sleep(30)
    
if __name__=="__main__":
    main()