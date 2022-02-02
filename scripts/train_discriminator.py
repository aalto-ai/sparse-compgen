import argparse
import math
import os
import pickle

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

from babyaiutil.datasets.discriminator import (
    make_discriminator_dataset_from_trajectories,
    make_initial_observation_discriminator_dataset_from_trajectories,
)
from babyaiutil.models.discriminator.film import FiLMDiscriminatorHarness
from babyaiutil.models.discriminator.simple_attention import (
    SimpleAttentionDiscriminatorHarness,
)
from babyaiutil.models.discriminator.transformer import TransformerDiscriminatorHarness
from babyaiutil.models.discriminator.independent_attention import (
    IndependentAttentionDiscriminatorHarness,
)
from babyaiutil.callbacks.schedule_hparam import ScheduleHparamCallback


MODELS = {
    "film": FiLMDiscriminatorHarness,
    "transformer": TransformerDiscriminatorHarness,
    "independent": IndependentAttentionDiscriminatorHarness,
    "independent_noreg": IndependentAttentionDiscriminatorHarness,
    "simple": SimpleAttentionDiscriminatorHarness,
    "simple_noreg": SimpleAttentionDiscriminatorHarness,
}


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--model", choices=list(MODELS.keys())
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
    parser.add_argument("--iterations", default=50000, type=int)
    parser.add_argument(
        "--total", default=10000, type=int, help="Total number of instances per task"
    )
    parser.add_argument(
        "--batch-size", default=1024, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--check-val-every", default=20, type=int, help="Check val every N steps"
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

    train_dataset = make_discriminator_dataset_from_trajectories(
        train_trajectories, limit=effective_limit
    )
    valid_dataset_id = make_initial_observation_discriminator_dataset_from_trajectories(
        train_trajectories,
        limit=args.vlimit,
        offset=args.total - args.vlimit,
    )
    valid_dataset_ood = (
        make_initial_observation_discriminator_dataset_from_trajectories(
            valid_trajectories, limit=args.tlimit, offset=0
        )
    )

    pl.seed_everything(args.seed)
    model = MODELS[args.model](
        [0, len(OBJECT_TO_IDX), len(OBJECT_TO_IDX) + len(COLOR_TO_IDX)],
        48,
        len(words),
        lr=1e-4,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="vsf1/dataloader_idx_0",
        auto_insert_metric_name=False,
        save_top_k=5,
        mode="max",
    )
    callbacks = (
        [
            ScheduleHparamCallback("l1_penalty", 0, 10e-1, 1000, 5000),
        ]
        if args.model in ("independent", "simple")
        else []
    ) + [checkpoint_cb]

    trainer = pl.Trainer(
        max_steps=args.iterations,
        gpus=1,
        precision=16,
        default_root_dir=f"logs/{model_dir}/{exp_name}",
        callbacks=callbacks,
        val_check_interval=args.check_val_every,
    )
    pl.seed_everything(args.seed)
    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=args.batch_size),
        [
            DataLoader(valid_dataset_id, batch_size=len(valid_dataset_id)),
            DataLoader(valid_dataset_ood, batch_size=len(valid_dataset_ood)),
        ],
    )
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
