import argparse
import os
import pickle

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

from babyaiutil.datasets.discriminator import (
    make_discriminator_dataset_from_trajectories,
    make_initial_observation_discriminator_dataset_from_trajectories,
)
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
    parser.add_argument("--limit", default=None, type=int, help="Training set limit (per task)")
    parser.add_argument("--vlimit", default=None, type=int, help="Validation set limit (per task)")
    parser.add_argument("--tlimit", default=None, type=int, help="Test set limit (per task)")
    parser.add_argument("--iterations", default=50000, type=int)
    parser.add_argument("--total", default=10000, type=int, help="Total number of instances per task")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size for training")
    return parser


def do_experiment(args):
    effective_limit = min([args.limit or (args.total - args.vlimit), args.total - args.vlimit])

    exp_name = f"{args.exp_name}_s_{args.seed}_m_{args.model}_it_{args.iterations}_b_{args.batch_size}_l_{effective_limit}"
    print(exp_name)

    if os.path.exists(f"{exp_name}.pt"):
        print(f"Skipping {exp_name} as it already exists")
        return

    with open(args.data, "rb") as f:
        (train_trajectories, valid_trajectories, words, word2idx) = pickle.load(f)

    train_dataset = make_discriminator_dataset_from_trajectories(
        train_trajectories, limit=effective_limit
    )
    valid_dataset_id = make_initial_observation_discriminator_dataset_from_trajectories(
        train_trajectories, limit=args.vlimit, offset=args.total - args.vlimit,
    )
    valid_dataset_ood = make_initial_observation_discriminator_dataset_from_trajectories(
        valid_trajectories, limit=args.tlimit, offset=0
    )

    pl.seed_everything(args.seed)
    model = MODELS[args.model](
        [0, len(OBJECT_TO_IDX), len(OBJECT_TO_IDX) + len(COLOR_TO_IDX)],
        48,
        len(words),
        lr=1e-5,
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
        default_root_dir=f"logs/{args.exp_name}/{exp_name}",
        callbacks=callbacks,
    )
    pl.seed_everything(args.seed)
    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=args.batch_size),
        [
            DataLoader(valid_dataset_id, batch_size=len(valid_dataset_id)),
            DataLoader(valid_dataset_ood, batch_size=len(valid_dataset_ood)),
        ]
    )
    trainer.save_checkpoint(f"{exp_name}.pt")


def main():
    p = parser()
    args = p.parse_args()
    return do_experiment(args)


if __name__ == "__main__":
    main()
