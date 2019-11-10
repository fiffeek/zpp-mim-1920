from pretrain_bert import setup_parser
from wemux import TmuxSession
import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s]    %(message)s')


def run_bert_pretraining(args):
    session = TmuxSession(args.session_name)
    session.run_wemux_session()
    session.send_keys("sudo python3 {} --session-name {} --zone {} --tpu-name {} --project-name {} --bert-folder {} "
                      "--voc-size {} --vocab-thms-ls {} --vocab-filename {} --bert-config-filename {} "
                      "--checkpoints-steps {} --train-steps {} --tpu-cores {} --eval-batch-size {} "
                      "--train-batch-size {} --max-predictions {} --max-seq-length {} --masked-lm-prob {}"
                      "--bucket-name {} --model-dir {} --gcp-model-dir {} --pretraining-dir {}"
                      .format(args.executable, args.session_name, args.zone, args.tpu_name, args.project_name,
                              args.bert_folder, args.voc_size, args.vocab_thms_ls, args.vocab_filename,
                              args.bert_config_filename, args.checkpoints_steps, args.train_steps, args.tpu_cores,
                              args.eval_batch_size, args.train_batch_size, args.max_predictions, args.max_seq_length,
                              args.masked_lm_prob, args.bucket_name, args.model_dir, args.gcp_model_dir,
                              args.pretraining_dir))


setup_logging()
parser = setup_parser()
run_bert_pretraining(parser.parse_args())
