from pretrain_bert import setup_parser
from wemux import TmuxSession
import logging
import json


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s]    %(message)s')


def run_bert_pretraining(args):
    config_dump_filename = args.config_dump
    f = open(config_dump_filename, "a")
    f.close()
    f = open(config_dump_filename, "w")
    f.write(json.dumps(args.__dict__))

    session = TmuxSession(args.session_name)
    session.run_wemux_session()
    session.send_keys("sudo python3 {} --config-dump {}".format(args.executable, config_dump_filename))


setup_logging()
parser = setup_parser()
run_bert_pretraining(parser.parse_args())
