import sys
import os
import warnings

sys.path.append(os.path.dirname(__file__))
os.environ['NUMEXPR_MAX_THREADS'] = '8'
warnings.filterwarnings("ignore")

import time
import argparse
import json
from collections import OrderedDict
from models.DeepPCT import DeepPCTInferModel
from utils.data import Dataset
from utils.functional import get_project_root, get_sequence

project_root = get_project_root()

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input file')
arg_parser.add_argument('-o', '--output', type=str, default='output', help='Path to the output file')
arg_parser.add_argument('--jsonl', action='store_true', help='Output in JSONL format')
args = arg_parser.parse_args()

input_file_path = args.input
output_file_path = args.output


def check_input_and_output_file():
    global input_file_path, output_file_path
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f'Input file {input_file_path} does not exist.')

    if args.jsonl:
        if not output_file_path.endswith('.jsonl'):
            output_file_path += '.jsonl'
    else:
        if not output_file_path.endswith('.txt'):
            output_file_path += '.txt'

    assert not os.path.exists(
        output_file_path), f'Output file {output_file_path} already exists, please specify a new file name.'


def parse_input_file():
    with open(input_file_path, 'r') as f:
        input_lines = f.read().splitlines()

    data = OrderedDict()
    sequence_cache = {}
    for line in input_lines:
        seq_id, site1, site2 = line.split('\t')
        fasta_path = os.path.join(project_root, 'input/FASTA', f'{seq_id}.fasta')
        pdb_path = os.path.join(project_root, 'input/PDB', f'{seq_id}.pdb')
        site1_aa, site1_num = site1[0], int(site1[1:])
        site2_aa, site2_num = site2[0], int(site2[1:])

        sequence_cache[seq_id] = sequence_cache.get(seq_id, get_sequence(fasta_path))
        seq = sequence_cache[seq_id]
        assert seq[site1_num - 1] == site1_aa, \
            f'Site mismatch in protein {seq_id} at site {site1}, expected {site1_aa}, got {seq[site1_num - 1]} from ' \
            f'input FASTA file '
        assert seq[site2_num - 1] == site2_aa, \
            f'Site mismatch in protein {seq_id} at site {site2}, expected {site2_aa}, got {seq[site2_num - 1]} from ' \
            f'input FASTA file '

        if (fasta_path, pdb_path) in data:
            data[(fasta_path, pdb_path)].append((site1_num, site2_num))
        else:
            data[(fasta_path, pdb_path)] = [(site1_num, site2_num)]

    dataset = Dataset(data)
    return dataset


def run_prediction(dataset):
    model = DeepPCTInferModel(
        seq_weight_path=os.path.join(project_root, 'model_weights/DeepPCTseq.safetensors'),
        graph_weight_path=os.path.join(project_root, 'model_weights/DeepPCTgraph.safetensors'),
        site_weight_path=os.path.join(project_root, 'model_weights/DeepPCTsite.pkl'),
        esm2_650_weights_path=os.path.join(project_root, 'model_weights/esm2_t33_650M_UR50D.pt'),
        gearnet_edge_model_weight_path=os.path.join(project_root, 'model_weights/mc_gearnet_edge.pth')
    )

    probabilities = model(dataset)
    return probabilities


def write_output_file(dataset, probabilities):
    cut_off = 0.15

    if args.jsonl:
        # JSONL format: 
        # {"seq_id": "prefix of input FASTA and PDB file", "sites": [{"site": [site1, site2], "prediction_score": 0.123, "prediction_result": "Positive"}, ...]}
        output_lines = []
        index = 0
        for protein in dataset:
            protein_result = {
                'seq_id': protein.pdb_id,
                'sites': []
            }
            for site1_num, site2_num in protein:
                site1_aa, site2_aa = protein.sequence[site1_num - 1], protein.sequence[site2_num - 1]
                site1, site2 = f'{site1_aa}{site1_num}', f'{site2_aa}{site2_num}'
                prediction_score = probabilities[index]
                prediction_result = 'Positive' if prediction_score > cut_off else 'Negative'
                protein_result['sites'].append({
                    'site': [site1, site2],
                    'prediction_score': prediction_score,
                    'prediction_result': prediction_result
                })
                index += 1
            output_lines.append(json.dumps(protein_result))

    else:
        output_lines = []
        index = 0
        for protein in dataset:
            for site1_num, site2_num in protein:
                site1_aa, site2_aa = protein.sequence[site1_num - 1], protein.sequence[site2_num - 1]
                site1, site2 = f'{site1_aa}{site1_num}', f'{site2_aa}{site2_num}'
                prediction_score = probabilities[index]
                prediction_result = 'Positive' if prediction_score > cut_off else 'Negative'
                output_lines.append(
                    f'{protein.pdb_id}\t{site1}\t{site2}\tprediction_score\t{prediction_score:.3f}\tprediction_result\t{prediction_result}')
                index += 1

    with open(output_file_path, 'w') as f:
        f.write('\n'.join(output_lines))


def main():
    check_input_and_output_file()
    dataset = parse_input_file()
    probabilities = run_prediction(dataset)
    write_output_file(dataset, probabilities)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print(f'Prediction result has been written to {output_file_path}, time elapsed: {time_end - time_start:.2f}s')
