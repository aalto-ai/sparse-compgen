import argparse
import os
import pickle

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

from babyaiutil.datasets.discriminator import (
    make_discriminator_dataset_from_trajectories,
    make_path_discriminator_dataset_from_trajectories,
)
from babyaiutil.preprocess import mission_groups_indices
from babyaiutil.models.discriminator.film import FiLMDiscriminatorHarness
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
}


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--model", choices=["film", "transformer", "independent", "independent_noreg"]
    )
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--iterations", default=200000, type=int)
    return parser


def do_experiment(args):
    exp_name = f"{args.exp_name}_s_{args.seed}_m_{args.model}_it_{args.iterations}_l_{args.limit or 10000}"
    print(exp_name)

    if os.path.exists(f"{exp_name}.pt"):
        print(f"Skipping {exp_name} as it already exists")
        return

    with open(args.data, "rb") as f:
        (train_trajectories, valid_trajectories, words, word2idx) = pickle.load(f)

    train_dataset = make_discriminator_dataset_from_trajectories(
        train_trajectories, limit=args.limit
    )
    valid_dataset = make_path_discriminator_dataset_from_trajectories(
        valid_trajectories
    )

    pl.seed_everything(args.seed)
    model = MODELS[args.model](
        [0, len(OBJECT_TO_IDX), len(OBJECT_TO_IDX) + len(COLOR_TO_IDX)],
        32,
        len(words),
        lr=5e-4,
    )
    callbacks = (
        [
            ScheduleHparamCallback("l1_penalty", 0, 10e-1, 1000, 5000),
        ]
        if args.model == "independent"
        else []
    )

    trainer = pl.Trainer(
        max_steps=args.iterations,
        val_check_interval=20,
        gpus=1,
        default_root_dir=f"logs/{exp_name}",
        callbacks=callbacks,
    )
    pl.seed_everything(args.seed)
    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=128),
        DataLoader(valid_dataset, batch_size=1000),
    )
    trainer.save_checkpoint(f"{exp_name}.pt")


def main():
    p = parser()
    args = parser.parse_args()
    return do_experiment(args)


if __name__ == "__main__":
    main()
