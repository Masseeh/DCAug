# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

import argparse
import copy, os
import hashlib
import json
from pathlib import Path
import json

import numpy as np

from domainbed.datasets import num_environments 

import itertools

import shlex

import subprocess

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    myenv = os.environ.copy()
    for cmd in commands:
        subprocess.call(cmd, shell=True, env=myenv)

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, name):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest() + '_' + name.replace('_', '')

        self.train_args = copy.deepcopy(train_args)
        command = ['python', 'train_all.py', '--deterministic', args_hash]
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        # path setup
        self.log_dir = Path(train_args['log_dir']) / train_args['dataset'] / args_hash
        
        if (self.log_dir / 'done').exists():
            self.state = Job.DONE
        elif self.log_dir.exists():
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'])
        return '{}: {} {}'.format(
            self.state,
            self.log_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

def hparam_space(dataset, algorithm, space=1):
    SMALL_IMAGES = ["Debug28", "RotatedMNIST", "ColoredMNIST"]

    if space == 1:
        parameters = {
            'weight_decay': [1e-4, 1e-6],
            }

        if dataset in SMALL_IMAGES:
            parameters['lr'] = [3e-4, 1e-3, 3e-3]
        
        else:
            parameters['lr'] = [3e-4, 1e-5, 3e-5]
            parameters['resnet_dropout'] = [0.0, 0.1, 0.5],

        if algorithm == 'Augrino':
            parameters['aug_reg'] = [0.01, 0.05, 0.1]
    
    elif space == 2:
        parameters = {
            'num_ops': [1, 2],
            'magnitude': [2, 6, 10, 14]
            }
    elif space == 3:
        parameters = {
            'dropout_ratio': [0.8, 0.2],
            'n_dim': [8, 16, 32],
            'aug_weight_decay': [1e-6, 1e-4]
            }
        parameters['lr'] = [3e-4]
        parameters['aug_lr'] = [3e-4]
        parameters['weight_decay'] = [1e-6]


    return parameters

def make_args_list(dataset_names, algorithms, steps,
    data_dir, holdout_fraction, test_env, output_dir, space, extra_hp):
    args_list = []
    for dataset in dataset_names:
        for algorithm in algorithms:
            all_test_envs = [
                [i] for i in range(num_environments(dataset))]
            
            if test_env != None:
                all_test_envs = all_test_envs[test_env]

            for test_envs in all_test_envs:
                hparam = hparam_space(dataset, algorithm, space=space)
                hparam_keys = list(hparam.keys())

                for params in itertools.product(*list(hparam.values())):
                    train_args = {}
                    train_args['dataset'] = dataset
                    train_args['algorithm'] = algorithm
                    train_args['test_envs'] = [test_envs]
                    train_args['holdout_fraction'] = holdout_fraction
                    train_args['data_dir'] = data_dir
                    train_args['trial_seed'] = 0
                    train_args['seed'] = 0
                    train_args['log_dir'] = output_dir


                    #### SPECIAL CASE HPs
                    for k, p in extra_hp.items():
                        train_args[k] = p

                    for k, p in zip(hparam_keys, params):
                        train_args[k] = p

                    if steps is not None:
                        train_args['steps'] = steps
                    args_list.append(train_args)
    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

DATASETS = ['RotatedMNIST']
ALGORITHMS = ['TeachAdv']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument("name", type=str)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=ALGORITHMS)
    parser.add_argument('--output_dir', type=str, default='../sweep')
    parser.add_argument('--data_dir', type=str, default="datadir/")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--test_env', type=int)
    parser.add_argument('--space', type=int, default=1)
    parser.add_argument('--hp', type=json.loads, default={})
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    args_list = make_args_list(
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        steps=args.steps,
        data_dir=args.data_dir,
        holdout_fraction=args.holdout_fraction,
        test_env=args.test_env,
        output_dir=args.output_dir,
        space=args.space,
        extra_hp=args.hp
    )

    jobs = [Job(train_args, args.name) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
    print(f'About to launch {len(to_launch)} jobs.')
    if not args.skip_confirmation:
        ask_for_confirmation()
    Job.launch(to_launch, local_launcher)
