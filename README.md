# Domain Generalization by Rejecting Extreme Augmentations (Accepted at WACV 2024)

Official PyTorch implementation of [Domain Generalization by Rejecting Extreme Augmentations](https://arxiv.org/pdf/2310.06670.pdf).

## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

### Environments

Environment details used for our study.

```
Python: 3.8.16
PyTorch: 1.11.0+cu113
Torchvision: 0.12.0+cu113
CUDA: 11.3
NumPy: 1.23.5
PIL: 9.4.0
```

## How to Run

`train_all.py` script conducts a leave-one-out cross-validation for a given target domain.

```sh
python train_all.py exp_name --dataset PACS --test_envs target_domain --data_dir /my/datasets/path
```
`train_seed.py` script is a similar to `train_all.py` but run for multiple seeds.

```sh
python train_seed.py exp_name --dataset PACS --test_envs target_domain --data_dir /my/datasets/path --trails 3
```

### Reproduce the results of the paper

We provide the instructions to reproduce the main results of the paper, Table 3.
Note that the difference in a detailed environment or uncontrolled randomness may bring a little different result from the paper.

- PACS, VLCS, OfficeHome, TerraIncognita

```sh

for DS in "PACS" "VLCS" "OfficeHome" "TerraIncognita"
do

for test_envs in 0 1 2 3
do
python train_seed.py domain configs/config_erm_domain.yaml  --algorithm "ERMDAdv" --dataset $DS --test_envs $test_envs --deterministic --trials 3
done

for test_envs in 0 1 2 3
do
python train_seed.py teach_label configs/config_erm_label.yaml  --algorithm "ERMAdv" --dataset $DS --test_envs $test_envs --deterministic --trials 3 --use_teacher True
done

for test_envs in 0 1 2 3
do
python train_seed.py ta_wider configs/config_erm_ta.yaml  --algorithm "ERM" --dataset $DS --test_envs $test_envs --deterministic --trials 3 --auto_da "uniform" --tf_range "wider" --da_mode "online"
done

done
```

- DomainNet

```sh
for test_envs in 0 1 2 3 4 5
do
python train_seed.py domain configs/config_erm_domain.yaml  --algorithm "ERMDAdv" --dataset DomainNet --test_envs $test_envs --deterministic --trials 3
done

for test_envs in 0 1 2 3 4 5
do
python train_seed.py teach_label configs/config_erm_label.yaml  --algorithm "ERMAdv" --dataset DomainNet --test_envs $test_envs --deterministic --trials 3 --use_teacher True
done

for test_envs in 0 1 2 3 4 5
do
python train_seed.py ta_wider configs/config_erm_ta.yaml  --algorithm "ERM" --dataset DomainNet --test_envs $test_envs --deterministic --trials 3 --auto_da "uniform" --tf_range "wider" --da_mode "online"
done
```

## License

This project includes some code from [DomainBed](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414), and [SWAD](https://github.com/khanrc/swad/tree/main) also MIT licensed.
