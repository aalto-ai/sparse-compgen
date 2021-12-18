import numpy as np


def correct_rotation(image, direction):
    return np.fliplr(np.rot90(image, direction)).copy()


def wrap_state(env, d):
    d["rendered"] = env.grid.render(32, env.agent_pos, env.agent_dir)
    d["image"] = correct_rotation(d["image"], d["direction"])
    d["target_mask"] = correct_rotation(d["target_mask"], d["direction"])
    return d


class EnvWrapper(object):
    def __init__(self, env, seed=None):
        super().__init__()
        self.env = env
        self.seed = seed

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = wrap_state(self.env, state)

        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        return wrap_state(self.env, state)

    def copy(self):
        return EnvWrapper(self.env.copy(), seed=self.seed)

    def gen_obs(self):
        return wrap_state(self.env, self.env.gen_obs())
