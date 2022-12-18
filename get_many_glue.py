import subprocess

def main():
    main_model_dirs=[
        # "/home/hrenduchinta/LM-pretraining/models/wikitext_bert_10_12_2022_10:29:11",
        # "/home/hrenduchinta/LM-pretraining/models/wikitext_bert_11_12_2022_08:56:31",
        # "/home/hrenduchinta/LM-pretraining/models/wikitext_bert_11_12_2022_17:38:38",
        # "/home/hrenduchinta/LM-pretraining/models/wikitext_bert_10_12_2022_19:01:08",
        # "/home/hrenduchinta/LM-pretraining/models/wikitext_bert_12_12_2022_06:01:04"
        # "/home/hrenduchinta/LM-pretraining/models/wikitext_bert_15_12_2022_19:10:58"
        "/home/hrenduchinta/LM-pretraining/models/wikitext_bert_17_12_2022_18:25:01",
        "/home/hrenduchinta/LM-pretraining/models/wikitext_bert_18_12_2022_06:36:35"
    ]
    steps=[
        list(range(10000, 110000, 10000)),
        list(range(10000, 110000, 10000))
    ]
    config_files=["config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt"]
    model_dirs=[]
    for i, main_model_dir in enumerate(main_model_dirs):
        for step in steps[i]:
            model_dirs.append(f"{main_model_dir}/step_{step}/")
            for f in config_files:
                subprocess.run(["cp", f"{main_model_dir}/{f}", f"{main_model_dir}/step_{step}/{f}"])
    for i, model_dir in enumerate(model_dirs):
        subprocess.Popen(
            f"nohup python3 get_glue_metrics.py --model_dir {model_dir} --main_process_port {50355+i} --visible_gpus {i%8} > ./gluelogs_{i}.txt",
            shell=True
        )
    
if __name__=="__main__":
    main()