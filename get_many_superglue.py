import subprocess

def main():
    main_model_dirs=[
        "/home/hrenduchinta/LM-pretraining/models/fl_bert_07_10_2022_05:55:01"
    ]
    steps=[
        [50000, 100000, 150000, 200000, 250000],
    ]
    config_files=["config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt"]
    model_dirs=[]
    do_saves=[]
    for i, main_model_dir in enumerate(main_model_dirs):
        for step in steps[i]:
            if step==2500000:
                do_saves.append(True)
            else:
                do_saves.append(False)
            model_dirs.append(f"{main_model_dir}/step_{step}/")
            for f in config_files:
                subprocess.run(["cp", f"{main_model_dir}/{f}", f"{main_model_dir}/step_{step}/{f}"])
    for i, model_dir in enumerate(model_dirs):
        if do_saves[i]:
            subprocess.Popen(
                f"nohup python3 get_superglue_metrics.py --model_dir {model_dir} --visible_gpus {(i+5)%8} --do_save > ./logs/supergluelogs_{i}.txt",
                shell=True
            )
        else:
            subprocess.Popen(
                f"nohup python3 get_superglue_metrics.py --model_dir {model_dir} --visible_gpus {(i+5)%8} > ./logs/supergluelogs_{i}.txt",
                shell=True
            )
    
if __name__=="__main__":
    main()