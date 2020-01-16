import tensorflow as tf
import argparse
import os
from tqdm import tqdm
import json


tf.enable_eager_execution()
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, required=True, help='Specify directory for input files')
parser.add_argument('-t', '--type', type=str, required=True, help='Specify train/eval')
parser.add_argument('-o', '--output', type=str, required=True, help='Specify output file')
args = parser.parse_args()


def get_files():
    path = os.path.join(args.directory, args.type, 'tf*')
    return tf.gfile.Glob(path)


def parse_example(serialized):
    return tf.parse_single_example(serialized, features={
            'goal': tf.FixedLenFeature((), tf.string, default_value=''),
            'goal_asl': tf.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True),
            'tactic': tf.FixedLenFeature((), tf.string, default_value=''),
            'tac_id': tf.FixedLenFeature((), tf.int64, default_value=-1),
            'thms': tf.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True),
            'thms_hard_negatives': tf.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True),
        })


def example_to_dict(parsed_example):
    dc = {}
    dc['goal'] = parsed_example['goal'].numpy().decode("utf-8")
    dc['tactic'] = parsed_example['tactic'].numpy().decode("utf-8")
    dc['tac_id'] = parsed_example['tac_id'].numpy().item()
    dc['thms'] = list(map(lambda bstr: bstr.decode("utf-8"), parsed_example['thms'].numpy().tolist()))
    dc['goal_asl'] = list(map(lambda bstr: bstr.decode("utf-8"), parsed_example['goal_asl'].numpy().tolist()))
    dc['thms_hard_negatives'] = list(map(lambda bstr: bstr.decode("utf-8"), parsed_example['thms_hard_negatives'].numpy().tolist()))
    return dc


def process_files():
    data = []
    c = 0
    for fn in tqdm(get_files()):
        for record in tf.python_io.tf_record_iterator(fn):
            parsed_record = parse_example(record)
            record_dict = example_to_dict(parsed_record)
            data.append(record_dict)
            c += 1
    return data, c


def save_to_json(data, filename):
    with open(filename, 'w+') as f:
        json.dump(data, f)


print("Starting parsing data...")
print("NOTE: Requires a lot of RAM because all records are stored in memory for the time being.")
data, n_records = process_files()
print("Parsed data")
print("Starting saving the data...")
save_to_json(data, args.output)
print("Finished successfully")
print(n_records)
