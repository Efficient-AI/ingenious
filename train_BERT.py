import os
import subprocess
from datetime import datetime

def main():
    now=datetime.now()
    timestamp=now.strftime("%d_%m_%Y_%H:%M:%S")
    log_dir="./logs/fl_bert_"+timestamp+"/"
    model_dir="./models/fl_bert_"+timestamp +"/"
    subset_dir="./subsets/fl_bert_"+timestamp+"/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(subset_dir, exist_ok=True)
    l=[
        "accelerate", "launch", "run_lm_with_subsets_heuristic1.py",
        "--preprocessed",
        "--log_dir", log_dir,
        "--subset_dir", subset_dir,
        "--load_data_from_disk",
        "--data_directory", "./bert_dataset_prepared",
        # "--hidden_size", "512",
        # "--num_hidden_layers", "8",
        # "--num_attention_heads", "8",
        # "--intermediate_size", "2048",
        "--tokenizer_name", "bert-base-uncased",
        "--vocab_size", "30522",
        "--preprocess_batch_size", "2000",
        "--per_device_train_batch_size", "128",
        "--per_device_eval_batch_size", "128",
        "--learning_rate", "1e-4",
        "--weight_decay" ,"0.01",
        "--max_train_steps", "250000",
        "--gradient_accumulation_steps", "1",
        "--num_warmup_steps", "10000",
        "--output_dir", model_dir,
        "--max_seq_length", "128",
        "--preprocessing_num_workers", "96",
        "--mlm_probability" ,"0.15",
        "--short_seq_prob", "0.1",
        "--nsp_probability", "0.5",
        "--subset_fraction", "0.25",
        "--select_every", "25000",
        "--partition_strategy", "random",
        "--layer_for_similarity_computation", "9",
        "--num_partitions", "5000",
        "--selection_strategy", "fl",
        "--parallel_processes", "96",
        "--num_warmstart_epochs", "0",
        "--checkpointing_steps", "25000",
        # "--knn_index_key", "IVF65536,flat",
        # "--knn_ngpu", "8",
        # "--knn_tempmem", "0",
        # "--knn_altadd", 
        # "--knn_use_float16",
        # "--knn_abs", "2097152",
        # "--knn_nprobe", "64",
        # "--knn_nnn", "80"
        # "--resume_from_checkpoint", "/home/sumbhati/ingenious/LM-pretraining/models/fl_bert_19_07_2022_21:42:20/step_50000/"
    ]
    with open(log_dir+"parameters.txt", "w") as f:
        for item in l:
            f.write(item)
            f.write("\n")
    subprocess.run(l)
    for i in range(1, 4):
        model_name_or_path=model_dir#+"step_{}/".format(100)
        tasks=["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "mnli", "qqp"] #can also add "mnli", "qnli", "qqp", "sst2" 
        glue_log_dir=model_name_or_path+f"glue_run_{i}/"
        os.makedirs(glue_log_dir, exist_ok=True)
        for task in tasks:
            l=[
                "accelerate", "launch", "run_glue.py",
                "--log_file", glue_log_dir+task+".log",
                "--task_name", task,
                "--max_length", "128",
                "--model_name_or_path", model_name_or_path,
                "--per_device_train_batch_size", "8",
                "--per_device_eval_batch_size", "8",
                "--learning_rate", "5e-5",
                "--weight_decay" ,"0.0",
                "--num_train_epochs", "3",
                # "--seed", "23",
            ]
            subprocess.run(l)

if __name__=="__main__":
    main()