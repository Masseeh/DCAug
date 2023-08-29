# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run seeds
"""

import argparse
import copy, os
import json
import numpy as np
from pathlib import Path
from prettytable import PrettyTable

import shlex
from domainbed.lib import misc

import subprocess

import numpy as np 
import scipy.stats as st

def compute_mean_std_and_conf_interval(accuracies, confidence=.95):
    n = len(accuracies)
    m, s, se = np.mean(accuracies, axis=0), np.std(results, axis=0), st.sem(accuracies, axis=0)
    h = se * st.t.ppf((1 + confidence) / 2., n-1) if n > 1 else 0
    return m, s, h

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    myenv = os.environ.copy()
    for cmd in commands:
        subprocess.call(cmd, shell=True, env=myenv)

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, left_argv, trial_seed, device):

        self.train_args = copy.deepcopy(train_args)
        self.trial_seed = trial_seed
        timestamp = misc.timestamp()

        # self.train_args['seed'] = trial_seed

        self.unique_name = f"{timestamp}_{self.train_args['name']}_{self.train_args['test_envs'][0]}_{self.trial_seed}"
        command = ['python', 'train_all.py', self.train_args['name'], self.train_args['configs'][0], '--deterministic', '--device', device,
                        '--trial_seed', str(self.trial_seed), '--unique_name', self.unique_name] + left_argv

        for k, v in sorted(self.train_args.items()):
            if k in ['name', 'configs']: continue
            if v == None: continue

            if not isinstance(v, bool):

                if isinstance(v, list):
                    v = ' '.join([str(v_) for v_ in v])
                elif isinstance(v, str):
                    v = shlex.quote(v)
                
                command.append(f'--{k} {v}')

            else:
                if v:
                    command.append(f'--{k}')

        self.command_str = ' '.join(command)

        # path setup
        self.log_dir = Path(train_args['log_dir']) / train_args['dataset'] / self.unique_name
        
        self.check_state()

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.trial_seed)
        return '{}: {} {}'.format(
            self.state,
            self.log_dir,
            job_info)

    def check_state(self):
        if (self.log_dir / 'done').exists():
            self.state = Job.DONE
        elif self.log_dir.exists():
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run seeds")
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--data_dir", type=str, default="datadir/")
    parser.add_argument("--log_dir", type=str, default="train_output")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--algorithm", type=str, default="AdvAugBaseline")
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of steps. Default is dataset-dependent."
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("--test_envs", type=int, nargs="+", default=[0])  # sketch in PACS
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--starting_seed", type=int, default=0)
    parser.add_argument("--model_save", action="store_true", help="Model save start step")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tb_freq", default=10)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument("--tb", action="store_true", help="Run w/ tensorboard")
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")
    parser.add_argument(
        "--evalmode",
        default="fast",
        help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",
    )
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")
    args, left_argv = parser.parse_known_args()

    args = vars(args)
    device = args['device']
    del args['device']
    trials = args['trials']
    del args['trials']
    starting_seed = args['starting_seed']
    del args['starting_seed']


    jobs = [Job(args, left_argv, trial_seed + starting_seed, device) for trial_seed in range(trials)]

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
    # ask_for_confirmation()
    Job.launch(to_launch, local_launcher)

    trials = []
    for j in jobs:
        j.check_state()
        if j.state == Job.DONE:
            with open( Path(j.log_dir) / 'results.json' )  as json_file:
                data = json.load(json_file)
                trials.append( (data['results'], data['ds_envs']) )
    

    if len(trials) > 0:
        results = np.concatenate([ list(r[0].values()) for r in trials ]).reshape(len(trials), -1, len(trials[0][1]) + 1)[:, :, 0:1]

        mean, std, _ = compute_mean_std_and_conf_interval(results)

        table = PrettyTable(["Selection"] + trials[0][1])
        for key, mm, ss in zip(trials[0][0].keys(), mean, std):
            row = [f"Mean: {m :.4%}, Std: {s :.4%}" for m, s in zip(mm, ss)]
            table.add_row([key] + row)

        print(table)

        with open( Path(jobs[-1].log_dir) / 'log.txt' , 'a+')  as log_file:
            log_file.write(str(table))
