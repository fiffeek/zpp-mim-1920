import argparse
from wemux import TmuxSession
from pretrain_bert import BertConfig
import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s]    %(message)s')


def setup_parser():
    parser_tmp = argparse.ArgumentParser()
    parser_tmp.add_argument("-s", "--session-name", default="bert_pretraining", help="Name of a tmux console")
    parser_tmp.add_argument("-x", "--executable", default="colab.py", help="Name of executable bert pretraining")
    parser_tmp.add_argument("-z", "--zone", default="europe-west4-a", help="tpu zone")
    parser_tmp.add_argument("-n", "--tpu-name", default="devshell-vm-33444283-d998-465d-89d5-99a3bee1b061", help="tpu name to run")
    parser_tmp.add_argument("-p", "--project-name", default="zpp-mim-1920", help="gcp project name/id in which you run tpu")
    parser_tmp.add_argument("-b", "--bert-folder", default="bert", help="a folder name to which bert was cloned")
    parser_tmp.add_argument("-v", "--voc-size", default=2000, help="vocabulary size for a tokenizer")
    parser_tmp.add_argument("--vocab-thms-ls", default="vocab_thms_ls.txt", help="vocab thms ls file path")
    parser_tmp.add_argument("--vocab-filename", default="vocab.txt", help="vocab file name")
    parser_tmp.add_argument("--bert-config-filename", default="bert_config.json", help="config for bert given by a file name")
    parser_tmp.add_argument("--checkpoints-steps", default=2500)
    parser_tmp.add_argument("--train-steps", default=1000000)
    parser_tmp.add_argument("--tpu-cores", default=8)
    parser_tmp.add_argument("--eval-batch-size", default=64)
    parser_tmp.add_argument("--train-batch-size", default=128)
    parser_tmp.add_argument("--max-predictions", default=20)
    parser_tmp.add_argument("--max-seq-length", default=512)
    parser_tmp.add_argument("--masked-lm-prob", default=0.8)
    parser_tmp.add_argument("--bucket-name", default="zpp-bucket-1920", help="a name of a bucket to get data from/to")
    parser_tmp.add_argument("--model-dir", default="bert_model")
    parser_tmp.add_argument("--gcp-model-dir", default="bert-bucket-golkarolka/bert_model")
    parser_tmp.add_argument("--pretraining-dir", default="bert-bucket-golkarolka/pretraining_data")
    return parser_tmp


def run_bert_pretraining(args):
    session = TmuxSession(args.session_name)
    session.run_wemux_session()
    session.send_keys("python3 {}".format(args.executable))
    bert_config = BertConfig(
        bert_folder=args.bert_folder,
        voc_size=args.voc_size,
        vocab_thms_file_path=args.vocab_thms_ls,
        vocab_filename=args.vocab_filename,
        max_seq_length=args.max_seq_length,
        masked_lm_prob=args.masked_lm_prob,
        max_predictions=args.max_predictions,
        pretraining_dir=args.pretraining_dir,
        bucket_name=args.bucket_name,
        model_dir=args.model_dir,
        gcp_model_dir=args.gcp_model_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        train_steps=args.train_steps,
        save_checkpoints_steps=args.checkpoints_steps,
        num_tpu_cores=args.tpu_cores,
        bert_config_file=args.bert_config_filename,
        use_tpu=True,
        tpu_name=args.tpu_name,
        tpu_zone=args.zone,
        project=args.project_name
    )


setup_logging()
parser = setup_parser()
run_bert_pretraining(parser.parse_args())
