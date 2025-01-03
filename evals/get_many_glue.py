import subprocess

def main():
    main_model_dirs=[
        "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_16_09_2022_18:15:29",
        "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_16_09_2022_18:15:29",
        "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_16_09_2022_18:15:29",
        "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_16_09_2022_18:15:29",
        "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_16_09_2022_18:15:29",
        "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_16_09_2022_18:15:29",
        "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_16_09_2022_18:15:29",
        "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_16_09_2022_18:15:29",
        "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_16_09_2022_18:15:29",
        # "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_16_09_2022_18:16:58",
        # "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_lyer_12_16_09_2022_17:54:00",
        # "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_22_09_2022_20.35.55",
        # "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_10subset_26_09_2022_11:53:07",
        # "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_15subset_26_09_2022_11:49:48",
        # "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_20subset_29_09_2022_10:37:50",
        # "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_30subset_29_09_2022_10:53:54",
        # "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_2000_18_09_2022_18:30:48",
        # "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_2500_20_09_2022_18:54:02",
        # "/data-mount-milaggar-disk/home/hrenduchinta/milan-ingenious-data/models/fl_bert_3000_22_09_2022_18:20:32"
    ]
    steps=[
        [250000],
        [250000],
        [250000],
        [250000],
        [250000],
        [250000],
        [250000],
        [250000],
        [250000],
    ]
    runs=list(range(8, 18))
    config_files=["config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt"]
    model_dirs=[]
    for i, main_model_dir in enumerate(main_model_dirs):
        for step in steps[i]:
            model_dirs.append(f"{main_model_dir}/step_{step}/")
            for f in config_files:
                subprocess.run(["cp", f"{main_model_dir}/{f}", f"{main_model_dir}/step_{step}/{f}"])
    for i, model_dir in enumerate(model_dirs):
        subprocess.Popen(
            f"nohup python3 get_glue_metrics.py --model_dir {model_dir} --main_process_port {53355+i} --visible_gpus {i%8} --run {runs[i]} > ./gluelogs_{i}.txt",
            shell=True
        )
    
if __name__=="__main__":
    main()