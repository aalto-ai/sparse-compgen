import numpy as np
from copy import deepcopy
from queue import PriorityQueue

from env_utils import correct_rotation


def replay_actions_for_images(env, actions):
    root = deepcopy(env)

    current_obs = root.env.gen_obs()
    current_obs_rotated = correct_rotation(
        current_obs["image"], current_obs["direction"]
    )

    images = [current_obs_rotated] + [env.step(act)[0]["image"] for act in actions][:-1]

    return np.stack(images)


def solve_astar(env):
    q = PriorityQueue()
    node = deepcopy(env)

    current_obs = node.env.gen_obs()
    current_obs_rotated = correct_rotation(
        current_obs["image"], current_obs["direction"]
    )
    agent_pos = np.argwhere(current_obs_rotated.astype(np.int)[:, :, 0] == 10)[0]
    current_obs_mask_rotated = correct_rotation(
        current_obs["target_mask"], current_obs["direction"]
    )
    goal_locations = np.argwhere(
        current_obs_mask_rotated.astype(np.int) == True  # noqa: E712
    )
    euc_distance = np.sqrt(
        ((goal_locations - agent_pos).astype(np.float) ** 2).sum(axis=1)
    ).min()

    # Things in the queue get sorted smallest item first,
    # so we add the distance and the path length so far
    #
    # To tiebreak, we compare rewards, so make them negative
    q.put((euc_distance, 0, euc_distance, [], [], [], node))

    seen = (
        np.repeat(np.ones_like(current_obs_mask_rotated)[:, :, None], 4, axis=2)
        * np.inf
    )

    while not q.empty():
        (
            priority,
            current_reward,
            last_distance,
            actions,
            locations,
            directions,
            node,
        ) = q.get()
        last_priority = priority - last_distance

        current_obs = node.env.gen_obs()
        current_obs_rotated = correct_rotation(
            current_obs["image"], current_obs["direction"]
        )
        agent_pos = np.argwhere(current_obs_rotated.astype(np.int)[:, :, 0] == 10)[0]
        agent_dir = current_obs["direction"]

        current_state = np.concatenate([agent_pos, [agent_dir]])

        seen[tuple(current_state)] = priority

        for act in range(7):
            child = deepcopy(node)
            next_state, reward, done, _ = child.step(act)

            if done:
                return (
                    current_reward + reward,
                    actions + [act],
                    np.stack(locations + [agent_pos]),
                    np.stack(directions + [agent_dir]),
                    replay_actions_for_images(env, actions + [act]),
                )
            else:
                next_agent_pos = np.argwhere(
                    next_state["image"].astype(np.int)[:, :, 0] == 10
                )[0]
                next_dir = next_state["direction"]
                next_target_mask = next_state["target_mask"]

                next_target_mask = np.argwhere(
                    next_target_mask.astype(np.int) == True  # noqa: E712
                )

                # We don't explore states where we picked up the
                # object, because then the goal location would disappear
                if not goal_locations.shape[0]:
                    continue

                euc_distance = np.sqrt(
                    ((goal_locations - next_agent_pos).astype(np.float) ** 2).sum(
                        axis=1
                    )
                ).min()
                candidate_priority = last_priority + 1 + euc_distance
                candidate_state = np.concatenate([next_agent_pos, [next_dir]])

                # Have we seen something better at this location? skip
                if seen[tuple(candidate_state)] < candidate_priority:
                    continue

                q.put(
                    (
                        candidate_priority,
                        current_reward - reward,
                        euc_distance,
                        actions + [act],
                        locations + [agent_pos],
                        directions + [agent_dir],
                        child,
                    )
                )
