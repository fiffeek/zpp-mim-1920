import os
import sys
import json
import nltk
import random
import logging
import tensorflow as tf
import sentencepiece as spm
from glob import glob
from google.colab import auth, drive
from tensorflow.keras.utils import Progbar
from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder


def append_bert():
    sys.path.append("bert")


def setup_logger():
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s :  %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    log.handlers = [sh]


USE_TPU = True

VOC_SIZE = 2000

bert_vocab = []

with open('vocab_thms_ls.txt', 'r') as f:
  for line in f:
    bert_vocab.append(line.rstrip('\n'))
    
ctrl_symbols = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]", "(", ")"]
bert_vocab = ctrl_symbols + bert_vocab

bert_vocab += ["[UNUSED_{}]".format(i) for i in range(VOC_SIZE - len(bert_vocab))]
print(len(bert_vocab))


VOC_FNAME = "vocab.txt" #@param {type:"string"}

with open(VOC_FNAME, "w") as fo:
  for token in bert_vocab:
    fo.write(token+"\n")


MAX_SEQ_LENGTH = 512 #@param {type:"integer"}
MASKED_LM_PROB = 0.15 #@param
MAX_PREDICTIONS = 20 #@param {type:"integer"}

PRETRAINING_DIR = "pretraining_data" #@param {type:"string"}

BUCKET_NAME = "zpp-bucket-1920"
MODEL_DIR = "bert_model"
tf.gfile.MkDir(MODEL_DIR)

bert_base_config = {
  "attention_probs_dropout_prob": 0.1, 
  "directionality": "bidi", 
  "hidden_act": "gelu", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 768, 
  "initializer_range": 0.02, 
  "intermediate_size": 3072, 
  "max_position_embeddings": 512, 
  "num_attention_heads": 12, 
  "num_hidden_layers": 12, 
  "pooler_fc_size": 768, 
  "pooler_num_attention_heads": 12, 
  "pooler_num_fc_layers": 3, 
  "pooler_size_per_head": 128, 
  "pooler_type": "first_token_transform", 
  "type_vocab_size": 2, 
  "vocab_size": VOC_SIZE
}

with open("{}/bert_config.json".format(MODEL_DIR), "w") as fo:
  json.dump(bert_base_config, fo, indent=2)
  
with open("{}/{}".format(MODEL_DIR, VOC_FNAME), "w") as fo:
  for token in bert_vocab:
    fo.write(token+"\n")


BUCKET_NAME = "zpp-bucket-1920"
MODEL_DIR = "bert-bucket-golkarolka/bert_model"
PRETRAINING_DIR = "bert-bucket-golkarolka/pretraining_data"
VOC_FNAME = "vocab.txt"

# Input data pipeline config
TRAIN_BATCH_SIZE = 128 #@param {type:"integer"}
MAX_PREDICTIONS = 20 #@param {type:"integer"}
MAX_SEQ_LENGTH = 512 #@param {type:"integer"}
MASKED_LM_PROB = 0.15 #@param

# Training procedure config
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
TRAIN_STEPS = 1000000 #@param {type:"integer"}
SAVE_CHECKPOINTS_STEPS = 2500 #@param {type:"integer"}
NUM_TPU_CORES = 8

if BUCKET_NAME:
  BUCKET_PATH = "gs://{}".format(BUCKET_NAME)
else:
  BUCKET_PATH = "."

BERT_GCS_DIR = "{}/{}".format(BUCKET_PATH, MODEL_DIR)
DATA_GCS_DIR = "{}/{}".format(BUCKET_PATH, PRETRAINING_DIR)

VOCAB_FILE = os.path.join(BERT_GCS_DIR, VOC_FNAME)
CONFIG_FILE = os.path.join(BERT_GCS_DIR, "bert_config.json")

INIT_CHECKPOINT = None; # tf.train.latest_checkpoint(BERT_GCS_DIR)

bert_config = modeling.BertConfig.from_json_file(CONFIG_FILE)
input_files = tf.gfile.Glob(os.path.join(DATA_GCS_DIR,'*tfrecord'))

log.info("Using checkpoint: {}".format(INIT_CHECKPOINT))
log.info("Using {} data shards".format(len(input_files)))

model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=INIT_CHECKPOINT,
      learning_rate=LEARNING_RATE,
      num_train_steps=TRAIN_STEPS,
      num_warmup_steps=10,
      use_tpu=USE_TPU,
      use_one_hot_embeddings=True)

tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      "devshell-vm-33444283-d998-465d-89d5-99a3bee1b061",
      zone="europe-west4-a",
      project="zpp-mim-1920")

run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=BERT_GCS_DIR,
      save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=SAVE_CHECKPOINTS_STEPS,
          num_shards=NUM_TPU_CORES,
          per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=USE_TPU,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE)
  
train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=MAX_SEQ_LENGTH,
        max_predictions_per_seq=MAX_PREDICTIONS,
        is_training=True)

estimator.train(input_fn=train_input_fn, max_steps=TRAIN_STEPS)


append_bert()
setup_logger()