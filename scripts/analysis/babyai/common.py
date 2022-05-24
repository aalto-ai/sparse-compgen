import collections
import itertools
import os
import numpy as np
import yaml


def generate_experiment_name(exp, model, seed, iterations, batch_size, limit):
    return f"{exp}_s_{seed}_m_{model}_it_{iterations}_b_{batch_size}_l_{limit}"


def get_most_recent_version(experiment_dir):
    versions = os.listdir(os.path.join(experiment_dir, "lightning_logs"))
    return sorted(versions, key=lambda x: int(x.split("_")[1]))[-1]


def get_best_k_models(logs_dir, task_name, model_name, experiment_name, version=None):
    experiment_dir = os.path.join(logs_dir, task_name, model_name, experiment_name)
    most_recent_version = get_most_recent_version(experiment_dir)
    directory = os.path.join(experiment_dir, "lightning_logs", most_recent_version)
    return os.path.join(directory, "checkpoints", "best_k_models.yaml")


def best_pair(items):
    try:
        return sorted(list(items), key=lambda x: x[1])[-1]
    except IndexError:
        return None


def get_path_to_best_scoring_model(best_k_models):
    try:
        with open(best_k_models, "r") as f:
            best = best_pair(yaml.load(f, yaml.FullLoader).items())
            return best
    except IOError:
        print(f"Skip {best_k_models}")
        return None


def get_checkpoint_scores_tuples_for_models(
    logs_dir, task_name, model_name, iterations, batch_size, seeds, limits
):
    return [
        (
            limit,
            sorted(
                [
                    (
                        seed,
                        get_path_to_best_scoring_model(
                            get_best_k_models(
                                logs_dir,
                                task_name,
                                model_name,
                                generate_experiment_name(
                                    task_name,
                                    model_name,
                                    seed,
                                    iterations,
                                    batch_size,
                                    limit,
                                ),
                            )
                        )[1],
                    )
                    for seed in seeds
                ],
                key=lambda x: -x[1],
            ),
        )
        for limit in limits
    ]


def seed_limit_pairs_to_model_paths(
    models_dir, task_name, model_name, iterations, batch_size, seed_limit_pairs
):
    return [
        os.path.join(
            models_dir,
            task_name,
            model_name,
            model_filename,
        )
        for model_filename in [
            f"{generate_experiment_name(task_name, model_name, seed, iterations, batch_size, limit)}.pt"
            for seed, limit in seed_limit_pairs
        ]
    ]


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def compute_deltas(scores):
    return [
        (
            limit,
            [(row[0][0], 0.0)]
            + [
                (second_seed, (second_score - first_score))
                for (first_seed, first_score), (second_seed, second_score) in pairwise(
                    row
                )
            ],
        )
        for limit, row in scores
    ]


def trim_row_range_percent(row, pct):
    # Assume that the row has been sorted by score
    seeds, scores = list(zip(*row))
    min_score = scores[-1]
    max_score = scores[0]

    cutoff = max_score - ((max_score - min_score) * pct)

    mask = np.array(scores) > cutoff
    return np.array(seeds)[mask]


def cut_indices_at_biggest_drop(indices, scores):
    return np.array(indices)[: np.absolute(np.array(scores)).argmax()]


def get_best_rows_indices(deltas):
    return [(limit, cut_indices_at_biggest_drop(*zip(*row))) for limit, row in deltas]


def trim_rows_indices_pcts(rows, limit_lowerbound=10, limit_upperbound=500, pct=0.3):
    return [
        (limit, trim_row_range_percent(row, pct))
        for limit, row in rows
        if limit >= limit_lowerbound and limit <= limit_upperbound
    ]


def flatten_tree(tree):
    for limit, seeds in tree:
        for seed in seeds:
            yield (seed, int(limit))


def to_lists(sequence):
    if isinstance(sequence, np.ndarray):
        return sequence.tolist()

    if isinstance(sequence, collections.abc.Mapping):
        return {k: to_lists(v) for k, v in sequence.items()}

    if isinstance(sequence, collections.abc.Sequence):
        return [to_lists(v) for v in sequence]

    return sequence
