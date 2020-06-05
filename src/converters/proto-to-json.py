# Import Type Annotations
from __future__ import print_function

from tqdm import tqdm
import tensorflow as tf
from google.protobuf import text_format
from google.protobuf.json_format import MessageToJson
import deephol_pb2
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, required=True)
parser.add_argument('-f', '--fulljson', type=bool, required=False)


def load_text_proto(filename):
    """Load a protobuf from a text format file.

    Args:
      filename: Name of the file to be read.
      proto_constructor: The constructor method for the proto object.
      description: Optional string describing the content of the proto file.

    Returns:
      A protobuf parsed from the text file.
    """
    proto = deephol_pb2.ProofLog()
    with tf.gfile.Open(filename) as f:
        text_format.MergeLines(f, proto)

    return proto


def parameter_conclusion_to_dict(parameter_conclusion):
    single_conclusion = dict()
    single_conclusion['conclusion'] = parameter_conclusion
    return single_conclusion


def parameter_to_dict(parameter):
    conclusions = []
    if 'theorems' in parameter:
        for theorem in parameter['theorems']:
            conclusions.append(parameter_conclusion_to_dict(theorem['conclusion']))
    if 'term' in parameter:
        conclusions.append(parameter_conclusion_to_dict(parameter['term']))
    if 'conv' in parameter:
        conclusions.append(parameter_conclusion_to_dict(parameter['conv']))
    parameter_trimmed = dict()
    parameter_trimmed['parameterType'] = parameter['parameterType']
    parameter_trimmed['conclusions'] = conclusions
    return parameter_trimmed


def subgoal_to_dict(subgoal):
    subgoal_trimmed = dict()
    subgoal_trimmed['conclusion'] = subgoal['conclusion']
    return subgoal_trimmed


def proof_to_dict(proof):
    proof_trimmed = dict()
    subgoals = []
    parameters = []
    if 'subgoals' in proof:
        for subgoal in proof['subgoals']:
            subgoals.append(subgoal_to_dict(subgoal))
    if 'parameters' in proof:
        for parameter in proof['parameters']:
            parameters.append(parameter_to_dict(parameter))
    proof_trimmed['result'] = proof['result']
    proof_trimmed['tactic'] = proof['tactic']
    proof_trimmed['parameters'] = parameters
    proof_trimmed['subgoals'] = subgoals
    return proof_trimmed


def node_to_dict(node):
    proofs = []
    for proof in node['proofs']:
        proofs.append(proof_to_dict(proof))
    node_trimmed = dict()
    node_trimmed['status'] = node['status']
    node_trimmed['goal'] = node['goal']['conclusion']
    node_trimmed['proofs'] = proofs
    return node_trimmed


def serialize(filepath, fullJson=False):
    proto = load_text_proto(filepath)
    proto_json = json.loads(MessageToJson(proto))
    if fullJson:
        return json.dumps(proto_json, indent=2)
    data = []
    for node in proto_json['nodes']:
        if 'goal' in node and 'status' in node:
            data.append(node_to_dict(node))
    return json.dumps(data, indent=2)


if __name__ == '__main__':
    args = parser.parse_args()
    path = args.directory
    full = args.fulljson
    for filename in tqdm(os.listdir(path)):
        if filename.endswith(".pbtxt"):
            json_relative_path = os.path.splitext(filename)[0] + '.json'
            json_path = os.path.join(path, json_relative_path)
            file = open(json_path, 'w+')
            file.truncate(0)
            file.write(serialize(os.path.join(path, filename), full))
            file.close()

