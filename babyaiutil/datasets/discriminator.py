import random

import numpy as np
from torch.utils.data import Dataset, IterableDataset
from ..preprocess import mission_groups_indices
from . import TuplesDataset


class DiscriminatorDataset(IterableDataset):
    def __init__(self, dataset, groups_indices, limit=None):
        super().__init__()
        self.dataset = dataset
        self.groups_indices = list(groups_indices.values())

        if limit:
            self.groups_indices = [idx[:limit] for idx in self.groups_indices]

    def __iter__(self):
        while True:
            # Sample two distinct goals from the groups
            left_indices, right_indices = random.sample(self.groups_indices, 2)

            # Sample a datapoint from each
            left_sample_idx_1, left_sample_idx_2 = random.sample(left_indices, 2)
            left_sample_1, left_sample_2 = (
                self.dataset[left_sample_idx_1],
                self.dataset[left_sample_idx_2],
            )
            right_sample = self.dataset[random.choice(right_indices)]

            # For the first goal, take the rewarding state
            rewarding_image_left_1 = left_sample_1[0]
            rewarding_direction_left_1 = left_sample_1[-2]
            rewarding_image_left_2 = left_sample_2[0]
            rewarding_direction_left_2 = left_sample_2[-2]
            target_mission = left_sample_1[4]

            # For the second goal, take the rewarding state
            right_sample_image = right_sample[0]
            right_sample_direction = right_sample[-2]

            # First yield the "true" example
            #
            # In this example we have two images/directions of the same goal
            # and the label 1
            yield (
                target_mission,
                rewarding_image_left_1,
                rewarding_direction_left_1,
                rewarding_image_left_2,
                rewarding_direction_left_2,
                1,
            )

            # Then yield the "false" example
            #
            # In this example we have two images/directions of different goals
            # and the label 0
            yield (
                target_mission,
                rewarding_image_left_1,
                rewarding_direction_left_1,
                right_sample_image,
                right_sample_direction,
                0,
            )


def make_discriminator_dataset_from_trajectories(trajectories, limit=None):
    return DiscriminatorDataset(
        TuplesDataset(
            trajectories["image_trajectories"][
                np.arange(trajectories["image_trajectories"].shape[0]),
                trajectories["rewarding_idxs"],
            ],
            trajectories["image_trajectories"],
            trajectories["trajectory_masks"],
            trajectories["direction_trajectories"],
            trajectories["missions"],
            trajectories["direction_trajectories"][
                np.arange(trajectories["image_trajectories"].shape[0]),
                trajectories["rewarding_idxs"],
            ],
            trajectories["targets"],
        ),
        mission_groups_indices(trajectories["missions"]),
        limit=limit,
    )


class PathDiscriminatorDataset(Dataset):
    def __init__(
        self, missions, images_paths, direction_paths, rewards, masks, targets
    ):
        super().__init__()
        missions_path = missions[:, None].repeat(images_paths.shape[1], axis=1)
        targets_path = np.stack(targets)[:, None].repeat(images_paths.shape[1], axis=1)

        bool_masks = masks.astype(np.bool)
        masked_missions = missions_path[bool_masks]
        masked_images = images_paths[bool_masks]
        masked_directions = direction_paths[bool_masks]
        masked_rewards = rewards[bool_masks]
        masked_targets = targets_path[bool_masks]

        self.missions = masked_missions
        self.images = masked_images
        self.directions = masked_directions
        self.rewards = masked_rewards
        self.targets = masked_targets

    def __len__(self):
        return len(self.missions)

    def __getitem__(self, i):
        return (
            self.missions[i],
            self.images[i],
            self.directions[i],
            self.rewards[i],
            self.targets[i],
        )


def make_path_discriminator_dataset_from_trajectories(trajectories):
    return PathDiscriminatorDataset(
        trajectories["missions"],
        trajectories["image_trajectories"],
        trajectories["direction_trajectories"],
        trajectories["rewards"],
        trajectories["trajectory_masks"],
        trajectories["targets"],
    )
