import os
import subprocess
import argparse

# def parse_args():
#     parser=argparse.ArgumentParser(description="superglue")
#     parser.add_argument(
#         "--model_dir",
#         type=str,
#         required=True,
#         help="Path to pytorch saved model"
#     )
#     parser.add_argument(
#         "--main_process_port",
#         type=str,
#         default=55555,
#         help="main process port for huggingface accelerate"
#     )
#     args=parser.parse_args()
#     return args

def main():
    # args=parse_args()
    # model_dir=args.model_dir
    # datasets=["lambada", "ptb_text_only", "enwik8", "lm1b"]
    # for dataset in datasets:
    #     l=[
    #         "accelerate", "launch", "--main_process_port", f"{args.main_process_port}", "run_eval_gpt2.py",
    #         "--dataset_name", f"{dataset}",
    #         "--model_name_or_path", model_dir,
    #         "--num_train_epochs", "0",            
    #     ]
    #     subprocess.run(l)
    # datasets=["wikitext-2-raw-v1","wikitext-103-raw-v1"]
    # for dataset in datasets:
    #     l=[
    #         "accelerate", "launch", "--main_process_port", f"{args.main_process_port}", "run_eval_gpt2.py",
    #         "--dataset_name", "wikitext",
    #         "--dataset_config_name", f"{dataset}",
    #         "--model_name_or_path", model_dir,
    #         "--num_train_epochs", "0",        
    #     ]
    #     subprocess.run(l)   
    # args=parse_args()
    # model_dir=args.model_dir
    cntr=0
    main_model_dirs=[
        "/home/hrenduchinta/LM-pretraining/models/gpt2_vanilla_torch",
        "/home/hrenduchinta/LM-pretraining/models/gpt2_random_fixed_13_01_2023_08.07.20",
        "/home/hrenduchinta/LM-pretraining/models/gpt2_uncertainty_sampling_13_01_2023_22.17.06",
        "/home/hrenduchinta/LM-pretraining/models/gpt2_ingenious_09_01_2023_10.22.38"
    ]
    for main_model_dir in main_model_dirs:
        for step in range(50000, 550000, 50000):
            # for file in ["config.json", "merges.txt", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.json"]:
            #     subprocess.run(["cp", f"{main_model_dir}/{file}", f"{main_model_dir}/step_{step}/{file}"])
            model_dir=f"{main_model_dir}/step_{step}"
            datasets=[
                # "lambada",
                "openwebtextval",
                # "ptb_text_only",
                # "enwik8",
                # "lm1b",
                # "wikitext-103-raw-v1"
            ]
            for dataset in datasets:
                l=[
                    "accelerate", "launch", "--main_process_port", f"{55055+cntr}", "run_eval_gpt2.py",
                    "--log_dir", f"{model_dir}",
                    "--preprocessed",
                    "--load_data_from_disk",
                    "--tokenizer_name", "gpt2",
                    "--data_directory", f"{dataset}_prepared_1024",
                    "--model_name_or_path", f"{model_dir}",
                    "--max_train_steps", "0",
                    "--per_device_eval_batch_size", "16"
                ]
                subprocess.run(l)
                cntr+=1

if __name__=="__main__":
    main()