import os
import subprocess

def main():
    model_dir="models/fl_25percent_layer9_warmup_bert_06_06_2022_20:58:36/"
    log_dir=model_dir
    for i in range(1):
        model_name_or_path=model_dir#+"step_{}/".format(100)
        tasks=["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "mnli", "qqp"] #can also add "mnli", "qnli", "qqp", "sst2" 
        glue_log_dir=model_name_or_path+"glue_run1/"
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
    model_dir="models/fl_25percent_layer9_warmup_bert_06_06_2022_20:58:36/"
    log_dir=model_dir
    for i in range(1):
        model_name_or_path=model_dir#+"step_{}/".format(100)
        tasks=["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "mnli", "qqp"] #can also add "mnli", "qnli", "qqp", "sst2" 
        glue_log_dir=model_name_or_path+"glue_run2/"
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
    model_dir="models/fl_25percent_layer9_warmup_bert_06_06_2022_20:58:36/"
    log_dir=model_dir
    for i in range(1):
        model_name_or_path=model_dir#+"step_{}/".format(100)
        tasks=["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "mnli", "qqp"] #can also add "mnli", "qnli", "qqp", "sst2" 
        glue_log_dir=model_name_or_path+"glue_run3/"
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