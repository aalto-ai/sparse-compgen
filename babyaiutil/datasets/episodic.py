import itertools
from multiprocessing import Process, Pipe

import gc
import gym

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from ..wrap_env import correct_state_rotations

import sys


def compute_returns(rewards, gamma=0.9):
    returns = np.zeros(rewards.shape[0])
    returns[-1] = rewards[-1]

    for i in reversed(range(0, len(rewards) - 1)):
        returns[i] = gamma * returns[i + 1] + rewards[i]

    return returns


def grouper(iterable, n, fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def act_in_envs(envs, actions, previous_dones):
    images = []
    directions = []
    rewards = []
    dones = []

    for env, prev_done, action in zip(envs, previous_dones, actions):
        # Already done, no need to act in the environment
        if prev_done:
            obs = correct_state_rotations(env.gen_obs())

            images.append(obs["image"])
            directions.append(obs["direction"])
            rewards.append(0)
            dones.append(True)
        else:
            # Not done, yet, act in the environment
            obs, reward, done, _ = env.step(action.item())
            reward = 1 if (reward > 0) else 0
            obs = correct_state_rotations(obs)

            images.append(obs["image"])
            directions.append(obs["direction"])
            rewards.append(reward)
            dones.append(done)

    return np.stack(images), np.array(directions), np.array(rewards), np.array(dones)


def print_top_n_stats(stats, n, idx):
    print(f"Memory snapshot at at {idx}")
    for stat in stats[:n]:
        print(f"{stat}")

    sys.stdout.flush()


def print_top_stats(stats):
    for stat in stats:
        print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
        for line in stat.traceback.format():
            print(line)


def worker(conn, env_name, index):
    env = None
    done = False
    obs = None
    reward = 0
    info = None

    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            # If already done, send what we had before
            if done:
                conn.send((obs, reward, done, info))
            else:
                obs, reward, done, info = env.step(data)
                obs = correct_state_rotations(obs)

                conn.send((obs, reward, done, info))
        elif cmd == "seed":
            env = gym.make(env_name)
            env.seed(data)
            np.random.seed(data)

        elif cmd == "reset":
            obs = correct_state_rotations(env.reset())
            done = False
            conn.send(obs)
        else:
            raise NotImplementedError


def create_proc_with_pipe(target_fn, args):
    local, remote = Pipe()
    p = Process(target=target_fn, args=(remote, *args))
    p.daemon = True
    p.start()
    remote.close()

    return p, local


class ParallelEnvMultiproc(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, env_name, n_envs):
        assert n_envs >= 1, "No environment given."

        self.env_name = env_name

        self.locals = []
        self.processes = []
        self.n_envs = n_envs

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))

        return [local.recv() for local in self.locals]

    def seed(self, seeds):
        for local, seed in zip(self.locals, seeds):
            local.send(("seed", seed))

    def step(self, actions):
        for local, action in zip(self.locals, actions):
            local.send(("step", action))

        # We might only interact with a subset
        return [local.recv() for local, _ in zip(self.locals, actions)]

    def render(self):
        raise NotImplementedError

    def shutdown(self):
        for p in self.processes:
            p.terminate()
        self.locals = []
        self.processes = []

    def reboot(self):
        for p in self.processes:
            p.terminate()

        self.locals = []
        self.processes = []
        for i in range(self.n_envs):
            p, local = create_proc_with_pipe(worker, args=(self.env_name, i))
            self.locals.append(local)
            self.processes.append(p)

    def __del__(self):
        self.shutdown()


class ParallelEnvSingle(gym.Env):
    """A specialization of ParallelEnv with one process."""

    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name)

    def reset(self):
        return [self.env.reset()]

    def seed(self, seeds):
        self.env.seed(seeds[0])

    def step(self, actions):
        return [self.env.step(actions[0])]

    def render(self):
        raise NotImplementedError

    def shutdown(self):
        pass

    def reboot(self):
        pass

    @property
    def n_envs(self):
        return 1


class ParallelEnv(gym.Env):
    """A wrapper for concurrent execution."""

    def __init__(self, env_name, n_envs):
        assert n_envs >= 1, "No environment given."

        if n_envs == 1:
            self.base = ParallelEnvSingle(env_name)
        else:
            self.base = ParallelEnvMultiproc(env_name, n_envs)

    def reset(self):
        return self.base.reset()

    def seed(self, seeds):
        return self.base.seed(seeds)

    def step(self, actions):
        return self.base.step(actions)

    def render(self):
        return self.base.render()

    def shutdown(self):
        return self.base.shutdown()

    def reboot(self):
        return self.base.reboot()

    @property
    def n_envs(self):
        return self.base.n_envs


def collect_experience_from_policy(
    parallel_env, policy_model, word2idx, seeds, device=None
):
    def generate_experiences():
        first_parameter = next(policy_model.parameters())
        device = getattr(first_parameter, "device", None) or "cpu"

        policy_model.eval()

        for seeds_batch in grouper(seeds, parallel_env.n_envs):
            remaining_seeds_batch = [
                int(seed) for seed in seeds_batch if seed is not None
            ]
            parallel_env.seed(remaining_seeds_batch)
            all_initial_obs = parallel_env.reset()[: len(remaining_seeds_batch)]

            all_images = torch.tensor(
                np.stack([initial_obs["image"] for initial_obs in all_initial_obs]),
                dtype=torch.long,
                device=device,
            )
            all_directions = torch.tensor(
                np.array([initial_obs["direction"] for initial_obs in all_initial_obs]),
                dtype=torch.long,
                device=device,
            )
            all_missions = torch.tensor(
                np.stack(
                    [
                        np.array([word2idx[w] for w in initial_obs["mission"].split()])
                        for initial_obs in all_initial_obs
                    ]
                ),
                dtype=torch.long,
                device=device,
            )
            all_dones = np.array([False for env in remaining_seeds_batch])

            recorded_rewards = []
            recorded_dones = []
            recorded_actions = []

            with torch.no_grad():
                while not all_dones.all():
                    # In parallel, get the logits from the model
                    all_act_logits, _ = policy_model(
                        (all_missions, all_images[:, None], all_directions[:, None])
                    )[:2]
                    all_act_logits = all_act_logits.reshape(all_act_logits.shape[0], -1)
                    all_actions = all_act_logits.max(dim=-1)[1].detach().cpu().numpy()

                    all_obs, all_rewards, all_dones, _ = list(
                        zip(*parallel_env.step(all_actions))
                    )
                    all_images = np.stack([o["image"] for o in all_obs])
                    all_directions = np.array([o["direction"] for o in all_obs])
                    all_rewards = np.array(all_rewards)
                    all_dones = np.array(all_dones)

                    recorded_rewards.append(all_rewards)
                    recorded_dones.append(all_dones)
                    recorded_actions.append(all_actions)

                    all_images = torch.tensor(
                        all_images, dtype=torch.long, device=device
                    )
                    all_directions = torch.tensor(
                        all_directions, dtype=torch.long, device=device
                    )

            # Finished acting in environment, iterate over the rewards
            # we can ignore the other stuff now
            rewarding_indices = np.stack(recorded_dones).argmax(axis=0)
            recorded_rewards = np.stack(recorded_rewards)

            successes = [r[i] for r, i in zip(recorded_rewards.T, rewarding_indices)]

            if False:
                for seed, success, acts in zip(
                    remaining_seeds_batch, successes, np.stack(recorded_actions).T
                ):
                    if not success:
                        temp = gym.make(parallel_env.env_name)
                        temp.seed(seed)
                        obs = temp.reset()
                        print(obs["mission"], acts)
                        import matplotlib.pyplot as plt

                        plt.imshow(temp.render())
                        plt.show()

            yield from successes

    return generate_experiences


class FuncIterableDataset(IterableDataset):
    def __init__(self, func, n):
        super().__init__()
        self.func = func
        self.iterable = None
        self.n = n

    def __len__(self):
        return self.n

    def __iter__(self):
        self.iterable = self.func()
        return self

    def __next__(self):
        return next(self.iterable)


class EpisodicDataset(Dataset):
    def __init__(self, env, act_in_env_func, seeds):
        super().__init__()
        self.env = env
        self.act_in_env_func = act_in_env_func
        self.seeds = seeds

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, i):
        return self.act_in_env_func(self.env, self.seeds[i])
