from torch.utils.data import Dataset

from ..envs.babyai.preprocess import mission_groups_indices


class GoalsDataset(Dataset):
    """A dataset of labelled goals that is guaranteed to sampled in a balanced order.

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
        images,
        missions,
        targets,
        offset=None,
        limit=None,
    ):
        super().__init__()

        self.groups_indices = list(groups_indices.values())
        self.images = images
        self.missions = missions
        self.targets = targets

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
                idx[offset : offset + limit] for idx in self.groups_indices
            ]

    def __len__(self):
        return len(self.groups_indices) * len(self.groups_indices[0])

    def __getitem__(self, i):
        goal = i % len(self.groups_indices)
        j = i // len(self.groups_indices)

        idx = self.groups_indices[goal][j]

        return (self.images[idx], self.missions[idx], self.targets[idx])


def make_supervised_goals_dataset_from_trajectories(
    trajectories, limit=None, offset=None
):
    return GoalsDataset(
        mission_groups_indices(trajectories["missions"]),
        # Always take the first image in the trajectory,
        # that way the agent is in a random position
        trajectories["image_trajectories"][:, 0],
        trajectories["missions"],
        trajectories["targets"],
        limit=limit,
        offset=offset,
    )
