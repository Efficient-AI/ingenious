python3 run_language_modeling.py \
    --load_data_from_disk \
    --data_directory ./bert_dataset\
    --validation_split_percentage 5\
    --tokenizer_name ./MyBertTokenizerFast\
    --vocab_size 30000\
    --preprocess_batch_size 1000\
    --per_device_train_batch_size 256\
    --per_device_eval_batch_size 256\
    --learning_rate 1e-4\
    --weight_decay 0.01\
    --max_train_steps 100000\
    --gradient_accumulation_steps 1\
    --num_warmup_steps 0\
    --output_dir MyBERT\
    --max_seq_length 128\
    --preprocessing_num_workers 8\
    --mlm_probability 0.15\
    --short_seq_prob 0.1\
    --nsp_probability 0.5
