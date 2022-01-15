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

            # There is a 1-in-(n_goals - 1 + 1) chance that we sample
            # a null-goal, in which case the negative example
            # will be a non-goal state of the goal trajectory
            sampling_null_goal = random.random() < (1 / len(self.groups_indices))

            # Sample a datapoint from each
            left_sample_idx_1, left_sample_idx_2 = random.sample(left_indices, 2)
            left_sample_1, left_sample_2 = (
                self.dataset[left_sample_idx_1],
                self.dataset[left_sample_idx_2],
            )
            right_sample_idx = random.choice(right_indices)
            right_sample = self.dataset[right_sample_idx]

            # For the first goal, take the rewarding state
            rewarding_image_left_1 = left_sample_1[0]
            rewarding_direction_left_1 = left_sample_1[-2]
            rewarding_image_left_2 = left_sample_2[0]
            rewarding_direction_left_2 = left_sample_2[-2]
            target_mission = left_sample_1[4]

            # For the second goal, take the rewarding state
            # or if sampling an null-goal, a random
            # sample of the masked_trajectory (minus the goal state)
            if sampling_null_goal:
                null_goal_path_idx = random.choice(
                    list(
                        range(
                            len(left_sample_1[2][left_sample_1[2].astype(np.bool)]) - 1
                        )
                    )
                )
                right_sample_image = left_sample_1[1][null_goal_path_idx]
                right_sample_direction = left_sample_1[3][null_goal_path_idx]
            else:
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


def resample_array_by_groups_indices(groups_indices, array):
    return np.stack(
        [
            array[groups_indices[i % len(groups_indices)][i // len(groups_indices)]]
            for i in range(len(groups_indices) * len(groups_indices[0]))
        ]
    )


class PathDiscriminatorDataset(Dataset):
    def __init__(
        self,
        groups_indices,
        missions,
        images_paths,
        direction_paths,
        rewards,
        masks,
        targets,
        limit=None,
        offset=None,
    ):
        super().__init__()
        groups_indices = list(groups_indices.values())

        # We assume that all the groups are balanced
        assert all(
            [len(groups_indices[0]) == len(indices) for indices in groups_indices]
        )
        assert len(missions) == len(groups_indices) * len(groups_indices[0])

        offset = offset or 0

        if limit:
            assert offset + limit <= len(groups_indices[0])
            groups_indices = [idx[offset : offset + limit] for idx in groups_indices]

        resampled_missions = resample_array_by_groups_indices(groups_indices, missions)
        resampled_images_paths = resample_array_by_groups_indices(
            groups_indices, images_paths
        )
        resampled_direction_paths = resample_array_by_groups_indices(
            groups_indices, direction_paths
        )
        resampled_rewards = resample_array_by_groups_indices(groups_indices, rewards)
        resampled_targets = resample_array_by_groups_indices(
            groups_indices, np.stack(targets)
        )
        resampled_masks = resample_array_by_groups_indices(groups_indices, masks)

        resampled_missions_path = resampled_missions[:, None].repeat(
            resampled_images_paths.shape[1], axis=1
        )
        resampled_targets_path = resampled_targets[:, None].repeat(
            resampled_images_paths.shape[1], axis=1
        )

        resampled_bool_masks = resampled_masks.astype(np.bool)
        masked_missions = resampled_missions_path[resampled_bool_masks]
        masked_images = resampled_images_paths[resampled_bool_masks]
        masked_directions = resampled_direction_paths[resampled_bool_masks]
        masked_rewards = resampled_rewards[resampled_bool_masks]
        masked_targets = resampled_targets_path[resampled_bool_masks]

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


def make_path_discriminator_dataset_from_trajectories(
    trajectories, limit=None, offset=None
):
    return PathDiscriminatorDataset(
        mission_groups_indices(trajectories["missions"]),
        trajectories["missions"],
        trajectories["image_trajectories"],
        trajectories["direction_trajectories"],
        trajectories["rewards"],
        trajectories["trajectory_masks"],
        trajectories["targets"],
        limit=limit,
        offset=offset,
    )


def make_initial_observation_discriminator_dataset_from_trajectories(
    trajectories, limit=None, offset=None
):
    return PathDiscriminatorDataset(
        mission_groups_indices(trajectories["missions"]),
        trajectories["missions"],
        trajectories["image_trajectories"][:, :1],
        trajectories["direction_trajectories"][:, :1],
        trajectories["rewards"][:, :1],
        trajectories["trajectory_masks"][:, :1],
        trajectories["targets"],
        limit=limit,
        offset=offset,
    )
