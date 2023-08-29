import numpy as np
import matplotlib.pyplot as plt
import os

def show_weights(ax, steps, weights, labels, title):
    ax.stackplot(steps, weights[:, steps],
             labels=labels, alpha=0.8)
    ax.set_xticks(steps)
    ax.set_title(title)
    ax.set_xlabel('Steps x100')
    ax.set_ylabel('Probability')

def show_prob_mag(ax, steps, x, labels, ylabel):
    for ix, o in enumerate(labels):
        if x[ix][0] != None:
            ax.plot(steps, x[ix][steps], label=o)
            ax.legend(loc='upper right')
            ax.set_xticks(steps)
            ax.set_xlabel('Steps')
            ax.set_ylabel(ylabel)

def AAugGumble(plot_data, path):
    ops = plot_data['operations']
    stage_0 = plot_data['sub_policy_0']['stage_0']
    stage_1 = plot_data['sub_policy_0']['stage_1']
    steps = np.linspace(0, len(stage_0['weight']) - 1, 10, endpoint=True, dtype=int)

    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    show_weights(ax[0], steps, np.array(stage_0['weight']).T, ops, 'TFs (1st stage)')
    show_weights(ax[1], steps, np.array(stage_1['weight']).T, ops, 'TFs (2st stage)')
    lines, labels = ax[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'upper right')
    plt.savefig(os.path.join(path, 'sub_policies.png'))

    for i, s in enumerate([stage_0, stage_1]):
        stage_prob = np.array(s['prob']).T
        stage_mag = np.array(s['mag']).T

        fig, ax = plt.subplots(1, 2, figsize=(16,8))

        show_prob_mag(ax[0], steps, stage_prob, ops, 'Probability')
        show_prob_mag(ax[1], steps, stage_mag, ops, 'Magnitude')

        fig.suptitle(f'stage_{i}')

    plt.savefig(os.path.join(path, 'ops.png'), bbox_inches='tight')


def AAug(plot_data, path):
    subs = list(plot_data.keys())[1:]
    weights = [plot_data[k]['weight'] for k in subs]
    steps = np.linspace(0, len(weights[0]) - 1, 10, endpoint=True, dtype=int)
    fig, ax = plt.subplots(1, 1, figsize=(16,8))
    show_weights(ax, steps, np.array(weights), subs, 'Sub Policies')
    lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'upper right')
    plt.savefig(os.path.join(path, 'sub_policies.png'))

    fig, ax = plt.subplots(len(subs), 4, figsize=(20, 10))
    ax = ax.flatten()
    ix = 0

    for s in subs:
        for k, v in plot_data[s].items():
            if k == 'weight':
                continue
            prob = np.array(v['prob'])[None, ...]
            mag = np.array(v['mag'])[None, ...]

            show_prob_mag(ax[ix], steps, prob, [k], 'P')
            show_prob_mag(ax[ix + 1], steps, mag, [k], 'M')
            ix += 2

    plt.savefig(os.path.join(path, 'ops.png'), bbox_inches='tight')