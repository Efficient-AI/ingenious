python3 run_tokenizer_training.py \
    --load_data_from_disk \
    --data_directory ./bert_dataset \
    --batch_size 10000 \
    --vocab_size 30000 \
    --output_dir ./MyBertTokenizerFast