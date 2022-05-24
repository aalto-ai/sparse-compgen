from collections import Counter
from functools import partial
import multiprocessing as mp

import babyai
import gym
import numpy as np
from tqdm.auto import tqdm

from .wrap_env import correct_rotation


def yield_seeds(env_name):
    env = gym.make(env_name)

    seed = 0
    while True:
        env.seed(seed)
        env.reset()

        seed += 1
        yield seed


def env_from_seed(seed, env=None):
    if env is None:
        env = "BabyAI-GoToLocal-v0"

    if isinstance(env, str):
        env = gym.make(env)

    np.random.seed(seed)
    env.seed(seed)
    env.reset()

    return env


def filter_excess_environments(seeds, env, counter, n_each, expected_len):
    for seed in seeds:
        if sum(counter.values()) >= expected_len:
            break

        env = env_from_seed(seed, env)

        if counter[env.gen_obs()["mission"]] >= n_each:
            continue

        yield seed


def solve_env(seed, env=None):
    env = env_from_seed(seed, env)

    b = babyai.bot.Bot(env, fully_observed=True)
    done = False
    actions = []

    while not done:
        actions.append(b.replan())
        obs, reward, done, _ = env.step(actions[-1])

    return seed, (reward > 0), actions


def map_to_solution_mp(seeds, env_name, num_procs=8):
    if num_procs < 2:
        yield from map_to_solution(seeds, env_name)

    with mp.Pool(processes=num_procs) as pool:
        for seed, solved, acts in pool.imap_unordered(
            partial(solve_env, env=env_name), seeds, chunksize=100
        ):
            if solved:
                yield (seed, acts)


def map_to_solution(seeds, env):
    for seed in seeds:
        seed, solved, acts = solve_env(seed, env)

        if solved:
            yield (seed, acts)


def filter_by_solution_length(seed_solution_stream, min_length):
    for seed, solution in seed_solution_stream:
        if len(solution) < min_length:
            continue

        yield seed, solution


def filter_excess_solutions(seeds_and_solutions, env, counter, n_each):
    for seed, solution in seeds_and_solutions:
        env = env_from_seed(seed, env)

        if counter[env.gen_obs()["mission"]] >= n_each:
            continue

        yield seed, solution


def pos_from_obs(obs):
    rotated_obs = correct_rotation(obs["image"], obs["direction"])
    agent_pos = np.argwhere(rotated_obs.astype(np.int)[:, :, 0] == 10)[0]

    return agent_pos


def replay_actions_for_data(env_name, word2idx, seed, actions):
    env = gym.make(env_name)
    np.random.seed(seed)
    env.seed(seed)
    env.reset()

    initial_obs = env.gen_obs()
    obs_array = [initial_obs]
    rewards = []

    for act in actions:
        obs, reward, done, _ = env.step(act)
        rewards.append(reward)

        if not done:
            obs_array.append(obs)

    obs_tuples_array = [
        (
            correct_rotation(obs["image"], obs["direction"]),
            obs["direction"],
            pos_from_obs(obs),
        )
        for obs in obs_array
    ]
    images, directions, positions = list(zip(*obs_tuples_array))

    mission_idxs = [word2idx[w] for w in initial_obs["mission"].split()]

    assert rewards[-1] >= 0

    return (
        seed,
        np.stack(positions),
        np.array(actions),
        np.array(directions),
        np.array(rewards),
        correct_rotation(initial_obs["image"], initial_obs["direction"]),
        correct_rotation(initial_obs["target_mask"], initial_obs["direction"]),
        np.stack(images),
        np.array(mission_idxs),
    )


def generate_seeds_and_action_trajectories(
    env_name, n_each, n_combinations, min_length, num_procs=8
):
    n_expected = n_each * n_combinations
    unique_environments_counter = Counter()

    seeds_and_solutions = []

    bar = tqdm(total=n_expected)

    try:
        for seed, solution in (
            # We have to filter again to deal with excess
            # solutions coming through on the multiprocessing side
            filter_excess_solutions(
                filter_by_solution_length(
                    map_to_solution_mp(
                        filter_excess_environments(
                            yield_seeds(env_name),
                            env_name,
                            unique_environments_counter,
                            n_each,
                            n_expected,
                        ),
                        env_name,
                        num_procs=num_procs,
                    ),
                    min_length,
                ),
                env_name,
                unique_environments_counter,
                n_each,
            )
        ):
            env = env_from_seed(seed, env_name)
            mission = env.gen_obs()["mission"]

            assert unique_environments_counter[mission] < n_each

            unique_environments_counter[mission] += 1
            seeds_and_solutions.append((seed, solution))

            print(seed, mission, list(map(lambda x: int(x), solution)))
            bar.update(1)
    except KeyboardInterrupt:
        pass

    return seeds_and_solutions, unique_environments_counter


def replay_actions_for_data_star(args):
    return replay_actions_for_data(*args)


def replay_actions_for_data_mp(env_name, word2idx, seeds_and_solutions, num_procs=8):
    with mp.Pool(processes=num_procs) as pool:
        yield from pool.imap_unordered(
            replay_actions_for_data_star,
            map(lambda x: (env_name, word2idx, *x), seeds_and_solutions),
            chunksize=100,
        )
