import argparse
import itertools
import os
import operator
import pickle

import numpy as np
import pandas as pd
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import pytorch_lightning as pl

from babyaiutil.datasets.episodic import (
    FuncIterableDataset,
    collect_experience_from_policy,
)
from babyaiutil.datasets.trajectory import make_trajectory_dataset_from_trajectories
from babyaiutil.models.imitation.baseline import ACModelImitationLearningHarness
from babyaiutil.models.imitation.conv_transformer import (
    ConvTransformerImitationLearningHarness,
)
from babyaiutil.models.imitation.pure_transformer import (
    PureTransformerImitationLearningHarness,
)


MODELS = {
    "film_lstm_policy": ACModelImitationLearningHarness,
    "conv_transformer": ConvTransformerImitationLearningHarness,
    "pure_transformer": PureTransformerImitationLearningHarness,
}


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--model", choices=["film_lstm_policy", "conv_transformer", "pure_transformer"]
    )
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--vlimit", default=20, type=int)
    parser.add_argument("--tlimit", default=40, type=int)
    parser.add_argument("--total", default=10000, type=int, help="Total number of instances per task")
    parser.add_argument("--iterations", default=200000, type=int)
    return parser


def interactive_dataloader_from_seeds(env_name, model, word2idx, dataset, batch_size):
    seeds = [
        dataset.seeds[i] for i in itertools.chain.from_iterable(dataset.groups_indices)
    ]
    dataloader_il = DataLoader(
        FuncIterableDataset(
            collect_experience_from_policy(
                parallel_env, model, word2idx, seeds, batch_size
            ),
            len(seeds),
        ),
        batch_size=batch_size,
    )

    return dataloader_il


def do_experiment(args):
    exp_name = f"{args.exp_name}_s_{args.seed}_m_{args.model}_it_{args.iterations}_l_{args.limit or 10000}"
    print(exp_name)

    if os.path.exists(f"{exp_name}.pt"):
        print(f"Skipping {exp_name} as it already exists")
        return

    with open(args.data, "rb") as f:
        print("Opened", f.name)
        (train_trajectories, valid_trajectories, words, word2idx) = np.load(f, allow_pickle=True)

    train_dataset = make_trajectory_dataset_from_trajectories(
        train_trajectories, limit=args.limit - args.vlimit
    )
    valid_dataset_id = make_trajectory_dataset_from_trajectories(
        train_trajectories, limit=args.vlimit, offset=args.total - args.vlimit
    )
    valid_dataset_ood = make_trajectory_dataset_from_trajectories(
        valid_trajectories, limit=args.tlimit, offset=0
    )

    pl.seed_everything(args.seed)
    model = MODELS[args.model](lr=1e-3)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_id_dataloader_il = interactive_dataloader_from_seeds(
        "BabyAI-GoToLocal-v0", model, word2idx, valid_dataset_id, 64
    )
    val_ood_dataloader_il = interactive_dataloader_from_seeds(
        "BabyAI-GoToLocal-v0", model, word2idx, valid_dataset_ood, 64
    )

    print(model)
    pl.seed_everything(args.seed)
    trainer = pl.Trainer(
        callbacks=[pl.callbacks.LearningRateMonitor()],
        max_steps=args.iterations,
        # Every 500 steps, regardless of how large the training dataloader is
        val_check_interval=min(1.0, 500 / len(train_dataloader)),
        gpus=1,
        default_root_dir=f"logs/{exp_name}",
        accumulate_grad_batches=1,
    )
    trainer.fit(model, train_dataloader, [val_id_dataloader_il, val_ood_dataloader_il])
    trainer.save_checkpoint(f"{exp_name}.pt")


def main():
    p = parser()
    args = p.parse_args()
    return do_experiment(args)


if __name__ == "__main__":
    main()
