#!/usr/bin/env bash

sudo python3 bert/run_classifier.py --data_dir=bert/fine-tune_data --bert_config_file=bert/model/bert_config.json --task_name=cola --vocab_file=bert/model/vocab.txt --output_dir=gs://zpp-bucket-1920/bert-bucket-golkarolka/bert_model/fine-tune/ --init_checkpoint=gs://zpp-bucket-1920/bert-bucket-golkarolka/bert_model/model.ckpt-142500 --do_lower_case=False --max_seq_length=512 --do_train=True --do_eval=True --use_tpu=True --tpu_name=zpp-holist1 --gcp_project=zpp-mim-1920 --tpu_zone=us-central1-f --num_tpu_cores 8 --save_checkpoints_steps 10000 > out.out 2>&1
