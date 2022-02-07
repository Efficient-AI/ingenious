python3 run_BERT_tokenizer_training.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --batch_size 1000 \
    --vocab_size 25000 \
    --output_dir ./myBERTTokenizerFast