import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import FastDataLoader

def accuracy_from_loader(algorithm, loader, weights, device, debug=False, domain=None):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0

    algorithm.eval()

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        if domain != None:
            y = torch.zeros(x.size(0), device=device, dtype=int).fill_(domain)
        else:
            y = batch["y"].to(device)

        with torch.no_grad():
            if domain != None:
                logits = algorithm.predict_domain(x)
            else:
                logits = algorithm.predict(x)

            loss = F.cross_entropy(logits, y).item()

        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if debug:
            break

    algorithm.train()

    acc = correct / total
    loss = losssum / total
    return acc, loss

def accuracy_from_loader2(algorithm, loader, weights, device, debug=False, domain=None):
    correct_label = 0
    losssum_label = 0.0

    correct_domain = 0
    losssum_domain = 0.0

    total = 0
    weights_offset = 0

    algorithm.eval()

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        if domain != None:
            d = torch.zeros(x.size(0), device=device, dtype=int).fill_(domain)

        with torch.no_grad():
            logits = algorithm.predict(x)
            if domain != None:
                d_logits = algorithm.predict_domain(x)

            loss = F.cross_entropy(logits, y).item()
            if domain != None:
                d_loss = F.cross_entropy(d_logits, d).item()

        B = len(x)
        losssum_label += loss * B

        if domain != None:
            losssum_domain += d_loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)

        if logits.size(1) == 1:
            correct_label += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct_label += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        
        if domain != None:
            if d_logits.size(1) == 1:
                correct_domain += (d_logits.gt(0).eq(d).float() * batch_weights).sum().item()
            else:
                correct_domain += (d_logits.argmax(1).eq(d).float() * batch_weights).sum().item()

        total += batch_weights.sum().item()

        if debug:
            break

    algorithm.train()

    acc_label = correct_label / total
    loss_label = losssum_label / total

    if domain != None:
        acc_domain = correct_domain / total
        loss_domain = losssum_domain / total
    else:
        acc_domain = 0
        loss_domain = 0.0

    return acc_label, loss_label, acc_domain, loss_domain

def accuracy(algorithm, loader_kwargs, weights, device, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return accuracy_from_loader2(algorithm, loader, weights, device, **kwargs)


class Evaluator:
    def __init__(
        self, train_envs, test_envs, eval_meta, logger, device, evalmode="fast", debug=False, target_env=None
    ):
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug
        self.device = device
        self.d_idx = {env:idx for idx, env in enumerate(train_envs)} # domain index at training time
        self.target_env = target_env

    def evaluate(self, algorithm, ret_domain=True, ret_losses=False):
        n_train_envs = len(self.train_envs)
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0

        if ret_domain:
            summaries["dm_acc"] = 0.0
            summaries["dm_loss"] = 0.0

        accuracies = {}
        losses = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue
            
            is_test = env_num in self.test_envs
            is_target = env_num == self.target_env
            acc, loss, acc_d, loss_d = accuracy(algorithm, loader_kwargs, weights, self.device, debug=self.debug, domain=self.d_idx[env_num] if (ret_domain and not is_test) else None)

            accuracies[name] = acc
            losses[name] = loss

            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
                    if ret_domain:
                        summaries["dm_loss"] += loss_d / n_train_envs
                        summaries["dm_acc"] += acc_d / n_train_envs

            elif is_target:
                summaries["test_" + inout] += acc 

        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries
