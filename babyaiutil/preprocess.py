import numpy as np


def rewards_to_returns(trajectory, gamma=0.9):
    # Normalize all the rewards to 1
    rewards = trajectory[4].astype(np.bool).astype(np.float)
    returns = np.zeros(rewards.shape[0])
    returns[-1] = rewards[-1]

    for i in reversed(range(0, len(rewards) - 1)):
        returns[i] = gamma * returns[i + 1] + rewards[i]

    return rewards, returns


def pad_longest(array, pad_val=0):
    lengths = [len(a) for a in array]
    max_len = max(lengths)
    pad = np.ones([max_len] + list(array[0].shape[1:]), dtype=array[0].dtype) * pad_val
    padded = np.stack(
        [np.concatenate([a, pad[l:]], axis=0) for a, l in zip(array, lengths)]
    )
    masks = np.stack(
        [np.concatenate([np.ones_like(a), pad[l:] * 0]) for a, l in zip(array, lengths)]
    )

    return padded, masks, np.array(lengths)


def pad_dataset(trajectories):
    (
        envs,
        solutions,
        actions,
        directions,
        rewards,
        returns,
        images,
        targets,
        images_path,
        mission_idxs,
    ) = list(zip(*trajectories))

    padded_solutions, _, __ = pad_longest(solutions)
    padded_actions, _, __ = pad_longest(actions, pad_val=-1)
    padded_directions, _, __ = pad_longest(directions)
    padded_returns, return_masks, return_lengths = pad_longest(returns)
    padded_rewards, _, __ = pad_longest(rewards)
    padded_missions, mission_masks, mission_lengths = pad_longest(
        mission_idxs, pad_val=0
    )
    padded_images_path, _, __ = pad_longest(images_path, pad_val=0)
    rewarding_idxs = np.array([r.astype(np.bool).argmax(axis=-1) for r in rewards])

    return {
        "envs": np.array(envs),
        "missions": padded_missions,
        "mission_masks": mission_masks,
        "image_trajectories": padded_images_path,
        "direction_trajectories": padded_directions,
        "action_trajectories": padded_actions,
        "rewarding_idxs": rewarding_idxs,
        "returns": padded_returns,
        "rewards": padded_rewards,
        "trajectory_masks": return_masks,
        "trajectory_lengths": return_lengths,
        "targets": np.stack(targets),
    }


def mission_groups_indices(missions):
    missions_groups_indices = {}
    for i, mission in enumerate(missions):
        mission_str = " ".join([str(x) for x in mission])
        if mission_str not in missions_groups_indices:
            missions_groups_indices[mission_str] = [i]
        else:
            missions_groups_indices[mission_str].append(i)

    return missions_groups_indices
