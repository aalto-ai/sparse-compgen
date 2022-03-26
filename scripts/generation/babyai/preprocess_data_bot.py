#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import os

import numpy as np

from babyaiutil.envs.babyai.preprocess import pad_dataset, rewards_to_returns


def mask_dict_of_arrays(dict_of_arrays, mask):
    return {k: v[mask] for k, v in dict_of_arrays.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot-trajectories", required=True)
    parser.add_argument("output")
    args = parser.parse_args()

    with open(args.bot_trajectories, "rb") as f:
        x = pickle.load(f)
        (trajectories, word2idx, words) = x

    trajectories = [
        (*t[:4], *rewards_to_returns(t, gamma=0.9), *t[5:]) for t in trajectories
    ]

    padded_trajectories = pad_dataset(trajectories)

    test_combinations = np.array(
        [
            [word2idx[w] for w in s.split()]
            for s in [
                "go to a red ball",
                "go to the red ball",
                "go to a green box",
                "go to the green box",
                "go to a blue key",
                "go to the blue key",
                "go to a yellow key",
                "go to the yellow key",
                "go to a grey box",
                "go to the grey box",
                "go to a purple ball",
                "go to the purple ball",
            ]
        ]
    )

    test_train_mask = np.array(
        [(test_combinations == t[-1][None]).all(axis=1).any() for t in trajectories]
    )
    padded_train_trajectories = mask_dict_of_arrays(
        padded_trajectories, ~test_train_mask
    )
    padded_valid_trajectories = mask_dict_of_arrays(
        padded_trajectories, test_train_mask
    )

    with open(args.output, "wb") as f:
        pickle.dump(
            (padded_train_trajectories, padded_valid_trajectories, words, word2idx), f
        )


if __name__ == "__main__":
    main()
