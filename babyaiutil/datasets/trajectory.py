from torch.utils.data import Dataset

from ..preprocess import mission_groups_indices


class TrajectoryDataset(Dataset):
    """A dataset of trajectories that is guaranteed to sample in a balanced order.

    Walking through this dataset in-order will sample one goal after
    the other and a trajectory from that goal. You can use this property to
    take a subset of the dataset up to some multiple of len(groups_indices)
    and get a balanced subset that way.

    You can also use the :limit: parameter to get a subset, which will
    be balanced, as it is a subset of each goal in the dataset.
    """

    def __init__(
        self,
        groups_indices,
        seeds,
        missions,
        images_paths,
        direction_paths,
        actions,
        returns,
        masks,
        offset=None,
        limit=None,
    ):
        super().__init__()

        self.groups_indices = list(groups_indices.values())
        self.seeds = seeds
        self.missions = missions
        self.images = images_paths
        self.directions = direction_paths
        self.actions = actions
        self.returns = returns
        self.masks = masks

        # We assume that all the groups are balanced
        assert all(
            [
                len(self.groups_indices[0]) == len(indices)
                for indices in self.groups_indices
            ]
        )
        assert len(self.missions) == len(self.groups_indices) * len(
            self.groups_indices[0]
        )

        offset = offset or 0

        if limit:
            assert offset + limit <= len(self.groups_indices[0])
            self.groups_indices = [
                idx[offset:offset + limit] for idx in self.groups_indices
            ]

    def __len__(self):
        return len(self.groups_indices) * len(self.groups_indices[0])

    def __getitem__(self, i):
        goal = i % len(self.groups_indices)
        j = i // len(self.groups_indices)

        idx = self.groups_indices[goal][j]

        return (
            self.seeds[idx],
            self.missions[idx],
            self.images[idx],
            self.directions[idx],
            self.actions[idx],
            self.returns[idx],
            self.masks[idx],
        )


def make_trajectory_dataset_from_trajectories(trajectories, limit=None, offset=None):
    return TrajectoryDataset(
        mission_groups_indices(trajectories["missions"]),
        trajectories["envs"],
        trajectories["missions"],
        trajectories["image_trajectories"],
        trajectories["direction_trajectories"],
        trajectories["action_trajectories"],
        trajectories["returns"],
        trajectories["trajectory_masks"],
        limit=limit,
        offset=offset,
    )
