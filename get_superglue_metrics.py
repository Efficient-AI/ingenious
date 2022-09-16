import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import os
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="superglue")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to pytorch saved model"
    )
    args=parser.parse_args()
    return args

def main():
    args=parse_args()
    tasks=[
        # "cola", "mnli", "mrpc", "qnli", "qqp", "sst", "stsb", "wnli", "rte",
        "boolq", "cb", "copa", "multirc", "wic", "wsc", #"record",
        # "boolq",
    ]
    model_dir=args.model_dir
    for task in tasks:
        for i in range(1, 21):
            jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
                task_config_base_path="./tasks/configs",
                task_cache_base_path="./cache",
                train_task_name_list=[task],
                val_task_name_list=[task],
                train_batch_size=8,
                eval_batch_size=8,
                epochs=10,
                num_gpus=8,
            ).create_config()
            os.makedirs(f"{model_dir}/superglue_run_{i}/run_configs/", exist_ok=True)
            py_io.write_json(jiant_run_config, f"{model_dir}/superglue_run_{i}/run_configs/{task}_run_config.json")
            display.show_json(jiant_run_config)

            run_args = main_runscript.RunConfiguration(
                jiant_task_container_config_path=f"{model_dir}/superglue_run_{i}/run_configs/{task}_run_config.json",
                output_dir=f"{model_dir}/superglue_run_{i}/runs/{task}",
                hf_pretrained_model_name_or_path=model_dir,
                model_path=f"{model_dir}/pytorch_model.bin",
                model_config_path=f"{model_dir}config.json",
                learning_rate=1e-5,
                adam_epsilon=1e-6,
                eval_every_steps=500,
                do_train=True,
                do_val=True,
                do_save=True,
                force_overwrite=True,
            )
            main_runscript.run_loop(run_args)

if __name__=="__main__":
    main()