import subprocess

def main():
    main_model_dirs=[
        "/home/hrenduchinta/LM-pretraining/models/gpt2_vanilla_torch",
        "/home/hrenduchinta/LM-pretraining/models/gpt2_ingenious_09_01_2023_10.22.38",
        "/home/hrenduchinta/LM-pretraining/models/gpt2_random_fixed_13_01_2023_08.07.20",
        "/home/hrenduchinta/LM-pretraining/models/gpt2_uncertainty_sampling_13_01_2023_22.17.06"
    ]
    steps=[
        list(range(50000, 550000, 50000)),
        list(range(50000, 550000, 50000)),
        list(range(50000, 550000, 50000)),
        list(range(50000, 550000, 50000))
    ]
    # steps=[
    #     [250000],
    #     [250000],
    #     [250000],
    #     [250000],
    # ]
    model_dirs=[]
    for i, main_model_dir in enumerate(main_model_dirs):
        for step in steps[i]:
            model_dirs.append(f"{main_model_dir}/step_{step}/")
    for task in ["gpt2_disambiguation_qa"]:
        for i, model_dir in enumerate(model_dirs):
            # subprocess.Popen(
            #     f"nohup python3 eval_gpt2_{task}.py --gpu {i%8} --model_name_or_path {model_dir} > ./gpt2_{task}_{i}.txt",
            #     shell=True
            # )
            subprocess.Popen(
                f"nohup python3 {task}.py --gpu {i%8} --model_name_or_path {model_dir} > ./gpt2_{task}_{i}.txt",
                shell=True
            )

if __name__=="__main__":
    main()