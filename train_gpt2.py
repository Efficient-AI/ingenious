import os
import subprocess
from datetime import datetime

def main():
    now=datetime.now()
    timestamp=now.strftime("%d_%m_%Y_%H:%M:%S")
    log_dir="./logs/gpt2_test_"+timestamp+"/"
    model_dir="./models/gpt2_test_"+timestamp +"/"
    os.makedirs(log_dir, exist_ok=True)
    l=[
        "accelerate", "launch", "run_clm.py",
        "--preprocessed",
        "--log_dir", log_dir,
        "--load_data_from_disk",
        "--data_directory", "./openwebtext_prepared_1024",
        "--config_name", "gpt2",
        "--tokenizer_name", "gpt2",
        "--per_device_train_batch_size", "8",
        "--per_device_eval_batch_size", "8",
        "--learning_rate", "1e-4",
        "--lr_max_steps", "1000000",
        "--weight_decay" ,"0.01",
        "--max_train_steps", "1000000",
        "--gradient_accumulation_steps", "2",
        "--num_warmup_steps", "10000",
        "--output_dir", model_dir,
        "--block_size", "1024",
        "--preprocessing_num_workers", "96",
        "--checkpointing_steps", "100000",
    ]
    with open(log_dir+"parameters.txt", "w") as f:
        for item in l:
            f.write(item)
            f.write("\n")
    subprocess.run(l)

if __name__=="__main__":
    main()