# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Example usage:
python -u -m domainbed.scripts.list_top_hparams \
    --input_dir domainbed/misc/test_sweep_data --algorithm ERM \
    --dataset VLCS --test_env 0
"""

from pathlib import Path

import argparse
import numpy as np

from domainbed.lib import reporting
from domainbed.lib import model_selection

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("name", type=str)
    parser.add_argument("--input_dir", type=str, default='../sweep')
    parser.add_argument('--dataset', default='RotatedMNIST')
    parser.add_argument('--algo', required=True)
    parser.add_argument('--test_env', type=int, required=True)
    args = parser.parse_args()

    in_dir = Path(args.input_dir) / args.dataset

    records = reporting.load_records(str(in_dir), args.name.replace('_', ''))
    print("Total records:", len(records))

    records = reporting.get_grouped_records(records)
    records = records.filter(
        lambda r:
            r['dataset'] == args.dataset and
            r['algorithm'] == args.algo and
            r['test_env'] == args.test_env
    )

    SELECTION_METHODS = [
        model_selection.IIDAccuracySelectionMethod,
        model_selection.OracleSelectionMethod,
    ]

    for selection_method in SELECTION_METHODS:
        print(f'Model selection: {selection_method.name}')

        for group in records:
            print(f"trial_seed: {group['trial_seed']}")
            best_hparams = selection_method.hparams_accs(group['records'])
            for run_acc, hparam_records in best_hparams[:1]:
                print(f"\t{run_acc}")
                for r in hparam_records:
                    assert(r['hparams'] == hparam_records[0]['hparams'])
                print("\t\thparams:")
                for k, v in sorted(hparam_records[0]['hparams'].items()):
                    print('\t\t\t{}: {}'.format(k, v))