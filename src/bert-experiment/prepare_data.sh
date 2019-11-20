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
python fine-tuning/prepare_data.py

# Directory for output files
mkdir bert/bert_output
