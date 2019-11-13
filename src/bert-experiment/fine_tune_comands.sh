#!/usr/bin/env bash

# Insert changes to bert model
gsutil cp gs://zpp-bucket-1920/bert-bucket-golkarolka/create_pretraining_data.py bert/
gsutil cp gs://zpp-bucket-1920/bert-bucket-golkarolka/tokenization.py bert/
gsutil cp gs://zpp-bucket-1920/bert-bucket-golkarolka/run_classifier.py bert/
mkdir bert/model

# Get pretrained model
gsutil cp gs://zpp-bucket-1920/bert-bucket-golkarolka/bert_model/model.ckpt-142500.data-00000-of-00001 bert/model
gsutil cp gs://zpp-bucket-1920/bert-bucket-golkarolka/bert_model/model.ckpt-142500.index bert/model
gsutil cp gs://zpp-bucket-1920/bert-bucket-golkarolka/bert_model/model.ckpt-142500.meta bert/model
# How to take always the newest version?

gsutil cp gs://zpp-bucket-1920/bert-bucket-golkarolka/bert_model/vocab.txt bert/model
gsutil cp gs://zpp-bucket-1920/bert-bucket-golkarolka/bert_model/bert_config.json bert/model

# Get and prepare data for fine-tuning
mkdir bert/fine-tune_data
gsutil cp gs://zpp-bucket-1920/bert-bucket-golkarolka/auto-tune.csv bert/fine-tune_data/
python fine-tuning/prepare_fine-tuning_data.py

# Directory for output files
mkdir bert/bert_output

python bert/run_classifier.py --data_dir=./bert/fine-tune_data --bert_config_file=./bert/model/bert_config.json --task_name=cola --vocab_file=./bert/model/vocab.txt --output_dir=./bert/bert_output --init_checkpoint=./bert/model/model.ckpt-142500.data-00000-of-00001 --do_lower_case=False --max_seq_length=512 --do_train=True --do_eval=True --use_tpu=True --tpu_name=grpc://10.125.162.34:8470 --tpu_zone=europe-west4-a --num_tpu_cores 8 --save_checkpoints_steps 10000