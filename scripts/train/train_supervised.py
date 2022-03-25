import argparse
import os
import pickle
import sys

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

from babyaiutil.datasets.supervised import (
    make_supervised_goals_dataset_from_trajectories,
)
from babyaiutil.models.supervised.film import FiLMConvEncoderProjectionHarness
from babyaiutil.models.supervised.transformer import (
    TransformerEncoderDecoderProjectionHarness,
)
from babyaiutil.models.supervised.independent_attention import (
    IndependentAttentionProjectionHarness,
)
from babyaiutil.callbacks.schedule_hparam import ScheduleHparamCallback


MODELS = {
    "film": FiLMConvEncoderProjectionHarness,
    "transformer": TransformerEncoderDecoderProjectionHarness,
    "independent": IndependentAttentionProjectionHarness,
    "independent_noreg": IndependentAttentionProjectionHarness,
}


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--model", choices=["film", "transformer", "independent", "independent_noreg"]
    )
    parser.add_argument(
        "--limit", default=None, type=int, help="Training set limit (per task)"
    )
    parser.add_argument(
        "--vlimit", default=None, type=int, help="Validation set limit (per task)"
    )
    parser.add_argument(
        "--tlimit", default=None, type=int, help="Test set limit (per task)"
    )
    parser.add_argument("--iterations", default=200000, type=int)
    parser.add_argument(
        "--total", default=10000, type=int, help="Total number of instances per task"
    )
    parser.add_argument(
        "--batch-size", default=1024, type=int, help="Batch size for training"
    )
    return parser


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

    with open(args.data, "rb") as f:
        (train_trajectories, valid_trajectories, words, word2idx) = pickle.load(f)

    train_dataset = make_supervised_goals_dataset_from_trajectories(
        train_trajectories, limit=effective_limit
    )
    valid_dataset_id = make_supervised_goals_dataset_from_trajectories(
        train_trajectories, limit=args.vlimit, offset=args.total - args.vlimit
    )
    valid_dataset_ood = make_supervised_goals_dataset_from_trajectories(
        valid_trajectories, limit=args.tlimit, offset=0
    )

    pl.seed_everything(args.seed)
    model = MODELS[args.model](
        [0, len(OBJECT_TO_IDX), len(OBJECT_TO_IDX) + len(COLOR_TO_IDX)],
        48,
        len(words),
        lr=1e-4,
    )
    callbacks = (
        [
            ScheduleHparamCallback("l1_penalty", 0, 10e-1, 1000, 5000),
        ]
        if args.model == "independent"
        else []
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    trainer = pl.Trainer(
        max_steps=args.iterations,
        # Every 20 steps, regardless of how large the training dataloader is
        val_check_interval=min(1.0, 20 / len(train_dataloader)),
        enable_progress_bar=sys.stdout.isatty(),
        gpus=1,
        default_root_dir=f"logs/{model_dir}/{exp_name}",
        callbacks=callbacks,
    )
    pl.seed_everything(args.seed)
    trainer.fit(
        model,
        train_dataloader,
        [
            DataLoader(valid_dataset_id, batch_size=len(valid_dataset_id)),
            DataLoader(valid_dataset_ood, batch_size=len(valid_dataset_ood)),
        ],
    )
    print(f"Done, saving {model_path}")
    trainer.save_checkpoint(f"{model_path}")


def main():
    p = parser()
    args = p.parse_args()
    return do_experiment(args)


if __name__ == "__main__":
    main()
