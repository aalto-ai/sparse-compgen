import collections
import os
import numpy as np


def generate_experiment_name(exp, model, seed, iterations, batch_size, limit):
    return f"{exp}_s_{seed}_m_{model}_it_{iterations}_b_{batch_size}_l_{limit}"


def get_most_recent_version(experiment_dir):
    versions = os.listdir(os.path.join(experiment_dir, "lightning_logs"))
    return sorted(versions, key=lambda x: int(x.split("_")[1]))[-1]


def to_lists(sequence):
    if isinstance(sequence, np.ndarray):
        return sequence.tolist()

    if isinstance(sequence, collections.abc.Mapping):
        return {k: to_lists(v) for k, v in sequence.items()}

    if isinstance(sequence, collections.abc.Sequence):
        return [to_lists(v) for v in sequence]

    return sequence
