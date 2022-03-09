python3 prepare_sample.py

# python3 run_tokenizer_training.py \
#     --load_data_from_disk \
#     --data_directory wikitext-2-raw-v1 \
#     --batch_size 1000 \
#     --vocab_size 25000 \
#     --output_dir MyBertTokenizerFast


accelerate launch run_language_modeling.py \
    --log_file ./logs/log_sample_bert_9thMar_0.log \
    --load_data_from_disk \
    --data_directory wikitext-2-raw-v1\
    --tokenizer_name bert-base-uncased\
    --vocab_size 30000\
    --preprocess_batch_size 1000\
    --per_device_train_batch_size 8\
    --per_device_eval_batch_size 8\
    --learning_rate 5e-5\
    --weight_decay 0.0\
    --num_train_epochs 3\
    --gradient_accumulation_steps 1\
    --num_warmup_steps 0\
    --output_dir ./models/sample_BERT_9thMar_0/\
    --max_seq_length 128\
    --preprocessing_num_workers 4\
    --mlm_probability 0.15\
    --short_seq_prob 0.1\
    --nsp_probability 0.5