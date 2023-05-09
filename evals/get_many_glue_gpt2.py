import subprocess

def main():
    main_model_dirs=[
        "/home/hrenduchinta/LM-pretraining/models/gpt2_random_fixed_13_01_2023_08.07.20"
    ]
    steps=[
        [50000, 450000]
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
            f"nohup python3 get_glue_metrics_gpt2.py --model_dir {model_dir} --main_process_port {55355+i} --visible_gpus {i%8} > ./gluelogs_{i}.txt",
            shell=True
        )
    
if __name__=="__main__":
    main()