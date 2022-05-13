import os
import subprocess

def main():
    model_dir="./models/BERT_10_05_2022_09:23:52/"
    log_dir=model_dir
    for i in range(1):
        model_name_or_path=model_dir+"model_checkpoint_{}/".format(8)
        tasks=["cola", "mrpc", "rte", "stsb", "wnli"] #can also add "mnli", "qnli", "qqp", "sst2" 
        glue_log_dir=model_name_or_path+"glue/"
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
            ]
            subprocess.run(l)

if __name__=="__main__":
    main()