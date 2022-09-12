import os
import subprocess
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="superglue")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to pytorch saved model"
    )
    parser.add_argument(
        "--main_process_port",
        type=str,
        required=False,
        help="main process port for huggingface accelerate"
    )
    args=parser.parse_args()
    return args

def main():
    args=parse_args()
    model_dir=args.model_dir
    log_dir=model_dir
    for i in range(1, 6):
        model_name_or_path=model_dir#+"step_{}/".format(100)
        tasks=["cola"]#, "mrpc", "rte", "stsb", "sst2", "qnli", "mnli", "qqp"] #can also add "mnli", "qnli", "qqp", "sst2" 
        glue_log_dir=model_name_or_path+f"test_glue_run_{i}/"
        os.makedirs(glue_log_dir, exist_ok=True)
        lrs=[2,3,4,5]
        for task in tasks:
            for lr in lrs:
                l=[
                    "accelerate", "launch", "--main_process_port", args.main_process_port, "run_glue.py",
                    "--log_file", glue_log_dir+task+f"_{lr}.log",
                    "--task_name", task,
                    "--max_length", "128",
                    "--model_name_or_path", model_name_or_path,
                    "--per_device_train_batch_size", "4",
                    "--per_device_eval_batch_size", "32",
                    "--learning_rate", f"{lr}e-5",
                    "--weight_decay" ,"0.0",
                    "--num_train_epochs", "3",
                    # "--seed", "45646",
                ]
                if args.main_process_port is None:
                    del l[2]
                    del l[2]
                subprocess.run(l)
    
if __name__=="__main__":
    main()