from __future__ import print_function

import hashlib
from tqdm import tqdm
import pandas as pd
import json
import argparse
import os

# generates single file from all jsons with proofs logs
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, required=True, help='Specify directory for input files')
parser.add_argument('-o', '--output', type=str, required=True, help='Specify single output file [will be in csv]')


PARAMETERS_TOKEN = 'EMPTY_TOKEN'


def proof_to_label(obj):
    if len(obj) > 0 and 'tactic' in obj[0]:
        return obj[0]['tactic']
    return 'FAILURE'


def proof_to_parameters(obj):
    if len(obj) > 0 \
            and 'parameters' in obj[0] \
            and len(obj[0]['parameters']) > 0 \
            and len(obj[0]['parameters'][0]['conclusions']) > 0:
        return obj[0]['parameters'][0]['conclusions']
    return PARAMETERS_TOKEN


def process_file(file_name, output):
    data = open(file_name, 'r').read()
    df_from_file = pd.DataFrame.from_dict(json.loads(data))
    df_bert = pd.DataFrame({'guid': df_from_file['goal'].map(lambda s: abs(hash(s)) % (10 ** 8)),
                            'label': df_from_file['proofs'].map(lambda proof: proof_to_label(proof)),
                            'parameters': df_from_file['proofs'].map(lambda proof: proof_to_parameters(proof)),
                            'theorem': df_from_file['goal']})

    df_bert.to_csv(output, sep=',', index=False, header=False, mode='a')


if __name__ == '__main__':
    args = parser.parse_args()
    path = args.directory
    output = args.output
    for filename in tqdm(os.listdir(path)):
        if filename.endswith(".json"):
            json_relative_path = os.path.splitext(filename)[0] + '.json'
            json_path = os.path.join(path, json_relative_path)
            process_file(json_path, output)
