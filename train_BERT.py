import subprocess
from datetime import datetime

def main():
    now=datetime.now()
    timestamp=now.strftime("%d_%m_%Y_%H:%M:%S")
    l=[
        "accelerate", "launch", "run_language_modeling.py",
        "--log_file", "./logs/log_bert_"+timestamp+".log",
        "--load_data_from_disk",
        "--data_directory", "./bert_dataset",
        "--tokenizer_name", "bert-base-uncased",
        "--vocab_size", "30522",
        "--preprocess_batch_size", "1000",
        "--per_device_train_batch_size", "32",
        "--per_device_eval_batch_size", "32",
        "--learning_rate", "1e-4",
        "--weight_decay" ,"0.01",
        "--max_train_steps", "1000000",
        "--gradient_accumulation_steps", "1",
        "--num_warmup_steps", "0",
        "--output_dir", "./models/BERT_"+timestamp +"/",
        "--max_seq_length", "128",
        "--preprocessing_num_workers", "32",
        "--mlm_probability" ,"0.15",
        "--short_seq_prob", "0.1",
        "--nsp_probability", "0.5",
    ]
    subprocess.run(l)

if __name__=="__main__":
    main()