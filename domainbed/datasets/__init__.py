import torch
import torchvision.transforms.functional as FT
import numpy as np

from domainbed.datasets import datasets
from domainbed.lib import misc
from domainbed.datasets import transforms as DBT

def num_environments(dataset_name):
    return len(datasets.get_dataset_class(dataset_name).ENVIRONMENTS)

def set_transfroms(dset, data_type, hparams, algorithm_class=None):
    """
    Args:
        data_type: ['train', 'valid', 'test', 'mnist']
    """

    additional_data = False
    if data_type == "train":
        dset.transforms = {"x": DBT.aug if hparams["data_augmentation"] == "aug" else DBT.basic}
        if hparams["num_ops"] >= 1:
            dset.append_transforms = {"x": DBT.MyRandAugment(num_ops=hparams["num_ops"], num_samples=hparams["num_samples"], strategy=hparams["auto_da"],
                                                                magnitude=hparams["magnitude"], rng=hparams["tf_range"])}
        additional_data = True
        dset.mode = hparams["da_mode"]
    elif data_type == "valid":
        dset.mode = "append"
        if hparams["val_augment"] is False:
            dset.transforms = {"x": DBT.basic}
        else:
            # Originally, DomainBed use same training augmentation policy to validation.
            # We turn off the augmentation for validation as default,
            # but left the option to reproducibility.
            dset.transforms = {"x": DBT.aug}
    elif data_type == "test":
        dset.mode = "append"
        dset.transforms = {"x": DBT.basic}
    elif data_type == "mnist":
        dset.mode = "append"
        # No augmentation for mnist
        dset.transforms = {"x": lambda x: x}
    else:
        raise ValueError(data_type)
    
    if additional_data and algorithm_class is not None:
        for key, transform in algorithm_class.transforms.items():
            dset.transforms[key] = transform


def get_dataset(test_envs, args, hparams, algorithm_class=None):
    """Get dataset and split."""
    is_mnist = "MNIST" in args.dataset
    dataset = vars(datasets)[args.dataset](args.data_dir)

    # if not is_mnist:
    #     dataset.input_shape = (3, 96, 96)

    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        # The split only depends on seed.
        # It means that the split is always identical only if use same seed,
        # independent to run the code where, when, or how many times.
        out, in_ = split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )
        if env_i in test_envs:
            in_type = "test"
            out_type = "test"
        else:
            in_type = "train"
            out_type = "valid"

        if is_mnist:
            in_type = "mnist"
            out_type = "mnist"

        set_transfroms(in_, in_type, hparams, algorithm_class)
        set_transfroms(out, out_type, hparams, algorithm_class)

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    return dataset, in_splits, out_splits


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transforms = {}
        self.append_transforms = {}
        self.mode = "append"

        self.direct_return = isinstance(underlying_dataset, _SplitDataset)

    def __getitem__(self, key):
        if self.direct_return:
            return self.underlying_dataset[self.keys[key]]

        x, y = self.underlying_dataset[self.keys[key]]
        ret = {"y": y}

        append_x = []
        for key, append_transform in self.append_transforms.items():
            append_x = append_transform(x)
            for i in range(len(append_x)):
                for key, transform in self.transforms.items():
                    append_x[i] = transform(append_x[i])

        if self.mode == "append":
            if len(append_x) > 1:

                for key, transform in self.transforms.items():
                    ret[key] = transform(x)

                ret["x"] = torch.stack([ret["x"], *append_x])

            else:
                for key, transform in self.transforms.items():
                    ret[key] = transform(x)

                if len(append_x) == 1:
                    ret["x"] = torch.stack([ret["x"], *append_x])
                    
        else:
            ret['x'] = append_x[0]

        return ret

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)
