import subprocess

def main():
    main_model_dirs=[
        "/data-mount-milaggar-disk/home/hrenduchinta/LM-pretraining/models/gpt2_vanilla_torch",
        "/data-mount-milaggar-disk/home/hrenduchinta/LM-pretraining/models/gpt2_ingenious_09_01_2023_10.22.38",
    ]
    steps=[
        [100000, 150000, 200000, 250000, 300000, 350000, 400000],
        [100000, 150000, 200000, 250000, 300000, 350000, 400000]
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
            f"nohup python3 get_glue_metrics_gpt2.py --model_dir {model_dir} --main_process_port {53355+i} --visible_gpus {(3+i)%8} > ./gpt2_gluelogs_{i}.txt",
            shell=True
        )
    
if __name__=="__main__":
    main()