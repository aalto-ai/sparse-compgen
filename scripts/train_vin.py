import argparse
import itertools
import json
import math
import os
import operator
import pickle
import sys

import babyai

import numpy as np
import pandas as pd
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

from babyaiutil.datasets.episodic import (
    FuncIterableDataset,
    ParallelEnv,
    collect_experience_from_policy,
)
from babyaiutil.datasets.trajectory import make_trajectory_dataset_from_trajectories
from babyaiutil.models.discriminator.independent_attention import (
    IndependentAttentionModel,
)
from babyaiutil.models.vin.harness import VINHarness

import tracemalloc


INTERACTION_MODEL = {"independent": IndependentAttentionModel}

MODELS = {"mvprop": VINHarness}


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", choices=["mvprop"])
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--vlimit", default=20, type=int)
    parser.add_argument("--tlimit", default=40, type=int)
    parser.add_argument(
        "--total", default=10000, type=int, help="Total number of instances per task"
    )
    parser.add_argument("--iterations", default=200000, type=int)
    parser.add_argument(
        "--n-eval-procs",
        default=4,
        type=int,
        help="Number of processes to run evaluation with",
    )
    parser.add_argument(
        "--batch-size", default=32, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--check-val-every", default=500, type=int, help="Check val every N steps"
    )
    parser.add_argument(
        "--interaction-model",
        choices=["independent"],
        default="independent",
        help="Which interaction module to use",
    )
    parser.add_argument(
        "--load-interaction-model",
        default=None,
        type=str,
        help="Path to an interaction model to load",
    )
    parser.add_argument(
        "--vin-k", default=10, type=int, help="Number of VIN iterations"
    )
    return parser


def interactive_dataloader_from_seeds(
    parallel_env, model, word2idx, dataset, batch_size
):
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


def filter_state_dict(state_dict, prefix):
    return {k.lstrip(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}


def memory_mapped_arrays(path):
    print("Memory-mapped archives in", path)
    with open(os.path.join(path, "contents.json"), "r") as f:
        contents = json.load(f)

    return {
        os.path.splitext(os.path.basename(k))[0]: np.memmap(
            os.path.join(path, k), dtype=dtype, shape=tuple(shape), mode="r"
        )
        for k, (dtype, shape) in contents.items()
    }


def load_json(path):
    print("Load json", path)
    with open(path, "r") as f:
        return json.load(f)


def load_data_directory(path):
    words = load_json(os.path.join(path, "words/words.json"))
    word2idx = load_json(os.path.join(path, "words/word2idx.json"))
    train_trajectories = memory_mapped_arrays(os.path.join(path, "train"))
    valid_trajectories = memory_mapped_arrays(os.path.join(path, "valid"))

    return (train_trajectories, valid_trajectories, words, word2idx)


def load_data_archive(path):
    with open(path, "rb") as f:
        print("Opened", f.name)
        return np.load(f, allow_pickle=True)


def load_data(path):
    if os.path.isdir(path):
        return load_data_directory(path)

    return load_data_archive(path)


def print_top_n_stats(stats, n, idx):
    print(f"Memory snapshot at at {idx}")
    for stat in stats[:n]:
        print(stat)


class MemoryStatsMonitor(pl.callbacks.base.Callback):
    def __init__(self, print_every, track_top_n):
        super().__init__()
        self.last_snapshot = None
        self.print_every = print_every
        self.track_top_n = track_top_n

    def on_train_start(self, trainer, pl_module):
        self.last_snapshot = None
        tracemalloc.start()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.print_every == 0:
            snapshot = tracemalloc.take_snapshot()
            if self.last_snapshot:
                print_top_n_stats(
                    snapshot.compare_to(self.last_snapshot, "lineno"), 20, batch_idx
                )
            else:
                print_top_n_stats(snapshot.statistics("lineno"), 20, batch_idx)

            self.last_snapshot = snapshot


def do_experiment(args):
    effective_limit = min(
        [args.limit or (args.total - args.vlimit), args.total - args.vlimit]
    )

    exp_name = f"{args.exp_name}_s_{args.seed}_m_{args.model}_it_{args.iterations}_b_{args.batch_size}_l_{effective_limit}"
    model_dir = f"models/{args.exp_name}/{args.model}"
    model_path = f"{model_dir}/{exp_name}.pt"
    print(model_path)

    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(f"{model_path}"):
        print(f"Skipping {model_path} as it already exists")
        return

    # We have to do some tricky things here since we're using multiprocessing
    #
    # In principle it shouldn't matter since Linux is copy-on-write. However,
    # doing things with python objects can sometimes increase reference counts,
    # which in turn copies the pages, which in turn increases memory usage.
    #
    # To ensure that we quarantine the worker processes from this huge chunk of
    # memory, do lets make sure to fork them off *before* we start loading big files
    # into memory.
    parallel_env = ParallelEnv("BabyAI-GoToLocal-v0", args.n_eval_procs)

    (train_trajectories, valid_trajectories, words, word2idx) = load_data(args.data)

    train_dataset = make_trajectory_dataset_from_trajectories(
        train_trajectories, limit=effective_limit
    )
    valid_dataset_id = make_trajectory_dataset_from_trajectories(
        train_trajectories, limit=args.vlimit, offset=args.total - args.vlimit
    )
    valid_dataset_ood = make_trajectory_dataset_from_trajectories(
        valid_trajectories, limit=args.tlimit, offset=0
    )

    pl.seed_everything(args.seed)
    offsets = [0, len(OBJECT_TO_IDX), len(OBJECT_TO_IDX) + len(COLOR_TO_IDX)]
    interaction_module = INTERACTION_MODEL[args.interaction_model](
        offsets, 48, len(words)
    )

    if args.load_interaction_model:
        interaction_module.load_state_dict(
            filter_state_dict(
                torch.load(args.load_interaction_model)["state_dict"], "model."
            )
        )
        print("Loaded interaction model", args.load_interaction_model)

    model = MODELS[args.model](interaction_module, offsets, 48, lr=1e-3, k=args.vin_k)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_id_dataloader_il = interactive_dataloader_from_seeds(
        parallel_env, model, word2idx, valid_dataset_id, 64
    )
    val_ood_dataloader_il = interactive_dataloader_from_seeds(
        parallel_env, model, word2idx, valid_dataset_ood, 64
    )

    check_val_opts = {}
    interval = args.check_val_every / len(train_dataloader)

    # Every check_val_interval steps, regardless of how large the training dataloader is
    if interval > 1.0:
        check_val_opts["check_val_every_n_epoch"] = math.floor(interval)
    else:
        check_val_opts["val_check_interval"] = interval

    print(model)
    print(check_val_opts)

    checkpoint_cb = ModelCheckpoint(
        monitor="vsucc/dataloader_idx_0",
        auto_insert_metric_name=False,
        save_top_k=5,
        mode="max",
    )

    pl.seed_everything(args.seed)
    trainer = pl.Trainer(
        callbacks=[pl.callbacks.LearningRateMonitor(), checkpoint_cb],
        max_steps=args.iterations,
        gpus=1,
        precision=16,
        default_root_dir=f"logs/{model_dir}/{exp_name}",
        accumulate_grad_batches=1,
        enable_progress_bar=sys.stdout.isatty(),
        **check_val_opts,
    )
    trainer.fit(model, train_dataloader, [val_id_dataloader_il, val_ood_dataloader_il])
    print(f"Done, saving {model_path}")
    trainer.save_checkpoint(f"{model_path}")
    print("Saving checkpoints info")
    checkpoint_cb.to_yaml()


def main():
    p = parser()
    args = p.parse_args()
    return do_experiment(args)


if __name__ == "__main__":
    main()
