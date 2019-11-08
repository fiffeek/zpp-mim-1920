from absl import flags
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

sys.path.append("bert")

from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder

auth.authenticate_user()

FLAGS = flags.FLAGS

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default='devshell-vm-33444283-d998-465d-89d5-99a3bee1b061',
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
flags.DEFINE_string(
    'gcp_project', default='zpp-mim-1920',
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone', default='europe-west4-a',
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu_name,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)
config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_shards,
          iterations_per_loop=FLAGS.iterations_per_loop))

sys.path.append("bert")

from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder

auth.authenticate_user()
  
# configure logging
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s :  %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
log.handlers = [sh]

if 'COLAB_TPU_ADDR' in os.environ:
  log.info("Using TPU runtime")
  USE_TPU = True
  TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']

  with tf.Session(TPU_ADDRESS) as session:
    log.info('TPU address is ' + TPU_ADDRESS)
    # Upload credentials to TPU.
    with open('/content/adc.json', 'r') as f:
      auth_info = json.load(f)
    tf.contrib.cloud.configure_gcs(session, credentials=auth_info)
    
else:
  log.warning('Not connected to TPU runtime')
  USE_TPU = False



AVAILABLE =  {'af','ar','bg','bn','br','bs','ca','cs',
              'da','de','el','en','eo','es','et','eu',
              'fa','fi','fr','gl','he','hi','hr','hu',
              'hy','id','is','it','ja','ka','kk','ko',
              'lt','lv','mk','ml','ms','nl','no','pl',
              'pt','pt_br','ro','ru','si','sk','sl','sq',
              'sr','sv','ta','te','th','tl','tr','uk',
              'ur','vi','ze_en','ze_zh','zh','zh_cn',
              'zh_en','zh_tw','zh_zh'}

LANG_CODE = "en" #@param {type:"string"}

assert LANG_CODE in AVAILABLE, "Invalid language code selected"

MODEL_PREFIX = "tokenizer" #@param {type: "string"}
VOC_SIZE = 2000 #@param {type:"integer"}
SUBSAMPLE_SIZE = 60000 #@param {type:"integer"}
NUM_PLACEHOLDERS = 256 #@param {type:"integer"}

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
DO_LOWER_CASE = False #@param {type:"boolean"}

PRETRAINING_DIR = "pretraining_data" #@param {type:"string"}
# controls how many parallel processes xargs can create
PROCESSES = 2 #@param {type:"integer"}
  


# XARGS_CMD = ("ls ./shards/ | "
#              "xargs -n 1 -P {} -I{} "
#              "python3 bert/create_pretraining_data.py "
#              "--input_file=./shards/{} "
#              "--output_file={}/{}.tfrecord "
#              "--vocab_file={} "
#              "--do_lower_case={} "
#              "--max_predictions_per_seq={} "
#              "--max_seq_length={} "
#              "--masked_lm_prob={} "
#              "--random_seed=34 "
#              "--dupe_factor=5")

# XARGS_CMD = XARGS_CMD.format(PROCESSES, '{}', '{}', PRETRAINING_DIR, '{}', 
#                              VOC_FNAME, DO_LOWER_CASE, 
#                              MAX_PREDICTIONS, MAX_SEQ_LENGTH, MASKED_LM_PROB)
                             
# tf.gfile.MkDir(PRETRAINING_DIR)
# !$XARGS_CMD

BUCKET_NAME = "bert-bucket-golkarolka" #@param {type:"string"}
MODEL_DIR = "bert_model" #@param {type:"string"}
tf.gfile.MkDir(MODEL_DIR)

if not BUCKET_NAME:
  log.warning("WARNING: BUCKET_NAME is not set. "
              "You will not be able to train the model.")

# use this for BERT-base

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


BUCKET_NAME = "bert-bucket-golkarolka" #@param {type:"string"}
MODEL_DIR = "bert_model" #@param {type:"string"}
PRETRAINING_DIR = "pretraining_data" #@param {type:"string"}
VOC_FNAME = "vocab.txt" #@param {type:"string"}

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

tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)

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


