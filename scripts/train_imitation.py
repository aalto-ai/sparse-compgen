import argparse
import itertools
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

from babyaiutil.datasets.episodic import (
    FuncIterableDataset,
    ParallelEnv,
    collect_experience_from_policy,
)
from babyaiutil.datasets.trajectory import make_trajectory_dataset_from_trajectories
from babyaiutil.models.imitation.baseline import ACModelImitationLearningHarness
from babyaiutil.models.imitation.conv_transformer import (
    ConvTransformerImitationLearningHarness,
)
from babyaiutil.models.imitation.sentence_encoder_image_decoder_transformer import (
    SentenceEncoderImageNSDecoderImageImitationLearningHarness,
)
from babyaiutil.models.imitation.fused_inputs_next_step_encoder_transformer import (
    FusedInputsNextStepTransformerEncoderHarness,
)
from babyaiutil.models.imitation.fused_inputs_autoregressive_transformer import (
    FusedInputsAutoregressiveTransformerHarness,
)


MODELS = {
    "film_lstm_policy": ACModelImitationLearningHarness,
    "conv_transformer": ConvTransformerImitationLearningHarness,
    "sentence_encoder_image_decoder": SentenceEncoderImageNSDecoderImageImitationLearningHarness,
    "fused_inputs_next_step_encoder": FusedInputsNextStepTransformerEncoderHarness,
    "fused_inputs_autoregressive_transformer": FusedInputsAutoregressiveTransformerHarness
}

ACCUM_SEQ_LENGTHS = {
    "fused_inputs_autoregressive_transformer": 16
}


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--model", choices=list(MODELS.keys())
    )
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
    return parser


def interactive_dataloader_from_seeds(
    parallel_env, model, word2idx, dataset, batch_size, max_seq_len=1, device=None
):
    seeds = [
        dataset.seeds[i] for i in itertools.chain.from_iterable(dataset.groups_indices)
    ]
    dataloader_il = DataLoader(
        FuncIterableDataset(
            collect_experience_from_policy(
                parallel_env, model, word2idx, seeds, max_seq_len=max_seq_len, device=device
            ),
            len(seeds),
        ),
        batch_size=batch_size,
    )

    return dataloader_il


def select_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


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

    with open(args.data, "rb") as f:
        print("Opened", f.name)
        (train_trajectories, valid_trajectories, words, word2idx) = np.load(
            f, allow_pickle=True
        )

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
    model = MODELS[args.model](lr=1e-3)
    validation_device = select_device()
    max_seq_len = ACCUM_SEQ_LENGTHS.get(args.model, 1)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_id_dataloader_il = interactive_dataloader_from_seeds(
        parallel_env, model, word2idx, valid_dataset_id, 64, max_seq_len=max_seq_len, device=validation_device
    )
    val_ood_dataloader_il = interactive_dataloader_from_seeds(
        parallel_env, model, word2idx, valid_dataset_ood, 64, max_seq_len=max_seq_len, device=validation_device
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
    pl.seed_everything(args.seed)
    trainer = pl.Trainer(
        callbacks=[pl.callbacks.LearningRateMonitor()],
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


def main():
    p = parser()
    args = p.parse_args()
    return do_experiment(args)


if __name__ == "__main__":
    main()
