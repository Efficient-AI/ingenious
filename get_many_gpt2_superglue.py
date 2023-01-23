import subprocess

def main():
    main_model_dirs=[
        # "/home/sumbhati/ingenious/LM-pretraining/models/gpt2_fl_19_09_2022_09.45.14",
        # "/home/sumbhati/ingenious/LM-pretraining/models/gpt2_03_09_2022_19.17.27",
        "/home/hrenduchinta/LM-pretraining/models/gpt2_ingenious_09_01_2023_10.22.38"
    ]
    steps=[
        list(range(50000, 550000, 50000))
    ]
    config_files=["config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]
    model_dirs=[]
    for i, main_model_dir in enumerate(main_model_dirs):
        for step in steps[i]:
            model_dirs.append(f"{main_model_dir}/step_{step}/")
            for f in config_files:
                subprocess.run(["cp", f"{main_model_dir}/{f}", f"{main_model_dir}/step_{step}/{f}"])
    for i, model_dir in enumerate(model_dirs):
        subprocess.Popen(
            f"nohup python3 get_gpt2_superglue_metrics.py --model_dir {model_dir} --main_process_port {51055+i} --visible_gpus {1+(i)%7} > ./gluelogs_{i}.txt",
            shell=True
        )
    
if __name__=="__main__":
    main()