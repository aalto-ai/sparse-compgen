import argparse
import collections
import fnmatch
import functools
import operator
import os
import itertools
import multiprocessing
import json

from tqdm.auto import tqdm, trange
import numpy as np
import pandas as pd
from rliable import library as rly
from rliable import metrics as rl_metrics
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from common import generate_experiment_name, get_most_recent_version, to_lists


DEFAULT_EVENTS = ["dataloader_idx_0/vsucc", "dataloader_idx_1/vsucc"]
DEFAULT_LIMITS = [50, 100, 250, 500, 1000, 2500, 5000, 9980]
DEFAULT_EXPERIMENTS = [
    "vin_sample_nogoal_16_logsumexp_ignorance:mvprop:70000:70000",
    "vin_sample_nogoal_16_logsumexp_k_0:mvprop:70000:70000",
    "imitation:pure_transformer:120000:70000",
    "imitation:film_lstm_policy:70000:70000",
]

# Experiments that didn't make it in the paper
OPTIONAL_EXPERIMENTS = [
    "vin_sample_nogoal_16_logsumexp_end_to_end_sparse:mvprop:270000:270000",
    "vin_sample_nogoal_16_logsumexp_ignorance_transformer:mvprop:70000:70000",
    "imitation:fused_inputs_next_step_encoder:250000:70000",
]


def get_tb_logs(logs_dir, task_name, model_name, experiment_name, version=None):
    experiment_dir = os.path.join(logs_dir, task_name, model_name, experiment_name)
    most_recent_version = get_most_recent_version(experiment_dir)
    directory = os.path.join(experiment_dir, "lightning_logs", most_recent_version)
    return os.path.join(
        directory, fnmatch.filter(os.listdir(directory), "*events.out*")[0]
    )


def tb_events_scalars_to_pd_dataframe(event_acc, events):
    event_acc.Reload()

    records = list(
        itertools.chain.from_iterable(
            [
                [
                    {"step": e.step, "value": e.value, "event": event}
                    for e in event_acc.Scalars(event)
                ]
                for event in events
            ]
        )
    )

    return pd.DataFrame.from_records(records)


def long_to_wide(df):
    return df.pivot_table(
        index=["event"], columns="step", values="value"
    ).T.reset_index()


def dataframe_for_experiment(
    task_name,
    model_name,
    experiment_name,
    logs_directory,
    models_directory,
    events=None,
    check_model=True,
):
    model_path = os.path.join(
        models_directory, task_name, model_name, f"{experiment_name}.pt"
    )
    if check_model and not os.path.exists(model_path):
        raise RuntimeError(f"{model_path} doesn't exist, experiment may be incomplete")

    df = tb_events_scalars_to_pd_dataframe(
        EventAccumulator(
            get_tb_logs(logs_directory, task_name, model_name, experiment_name)
        ),
        events=events or ["vmap", "vf1", "vtf1", "vtarget"],
    )
    return long_to_wide(df)


def average_dataframe(df):
    return df.mean(axis=0)


def average_over_seeds(
    exp,
    model,
    iterations,
    batch_size,
    limit,
    seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    events=None,
):
    df = pd.concat(
        [
            average_dataframe(
                dataframe_for_experiment(
                    exp,
                    model,
                    generate_experiment_name(
                        exp, model, seed, iterations, batch_size, limit
                    ),
                    events=events,
                )
            ).to_frame()
            for seed in seeds
        ],
        axis=1,
    )

    return df


def df_over_limit_range(
    exp,
    model,
    iterations,
    batch_size,
    limits=None,
    events=None,
):
    limits = limits or DEFAULT_LIMITS
    events = events or DEFAULT_EVENTS
    average_dfs = [
        average_over_seeds(
            exp, model, iterations, batch_size, limit, events=events
        ).T.assign(limit=limit)
        for limit in limits
    ]

    return pd.concat(
        [
            pd.melt(df.reset_index(), id_vars=["limit"], value_vars=events)
            for df in average_dfs
        ],
        axis=0,
    )


def drop_after(df, column, limit):
    return df[df[column] <= limit]


def keep_top(df, column, limit):
    return df.sort_values(column, ascending=False)[:limit]


def aggregate_func(x):
    return np.array([rl_metrics.aggregate_iqm(x[..., i]) for i in range(x.shape[-1])])


def generate_sample_efficiency_curves_plot_data(
    experiment_dfs, ylabel=None, xlabel=None
):
    aggregate_iqms, aggregate_cis = rly.get_interval_estimates(
        {k: df.transpose(1, 2, 0) for k, df in experiment_dfs.items()}, aggregate_func
    )

    return aggregate_iqms, aggregate_cis


def check_models_exist(
    experiment,
    model,
    models_directory,
    limits=None,
):
    expected_paths = list(
        itertools.chain.from_iterable(
            [
                [
                    (
                        os.path.join(
                            models_directory,
                            experiment,
                            model,
                            f"{generate_experiment_name(experiment, model, seed, iterations, batch_size, limit)}.pt",
                        )
                    )
                    for seed in range(10)
                ]
                for limit in limits
            ]
        )
    )
    paths_exist = dict({p: os.path.exists(p) for p in expected_paths})

    for p, e in paths_exist.items():
        if not e:
            print(f"Expected {p} to exist")

    if not all(paths_exist.values()):
        raise RuntimeError("Data incomplete!")


def sample_efficiency_matrix_configurations(limits=None):
    """Helper function to get the configurations for the sample efficiency matrices.

    Once we have the configurations, we can parallelize trivially.
    """
    return itertools.chain.from_iterable(
        [[(limit, seed) for seed in range(10)] for limit in limits]
    )


def generate_sample_efficiency_matrix_from_config(
    experiment,
    model,
    iterations,
    batch_size,
    keep_top_n_event,
    select_event,
    cut_at,
    models_directory,
    logs_directory,
    config,
):
    limit, seed = config

    df = dataframe_for_experiment(
        experiment,
        model,
        generate_experiment_name(
            experiment, model, seed, iterations, batch_size, limit
        ),
        models_directory=models_directory,
        logs_directory=logs_directory,
        events=[keep_top_n_event, select_event],
        check_model=False,
    )

    warning = (
        f"{generate_experiment_name(experiment, model, seed, iterations, batch_size, limit)} has only {df['step'].max()}"
        if df["step"].max() < 0.9 * cut_at
        else None
    )

    return (
        limit,
        seed,
        keep_top(drop_after(df, "step", cut_at), keep_top_n_event, 10)[
            select_event
        ].values,
        warning,
    )


def generate_sample_efficiency_matrix(
    experiment,
    model,
    iterations,
    batch_size,
    keep_top_n_event,
    select_event,
    cut_at,
    models_directory,
    logs_directory,
    limits=None,
    check_models=True,
):
    limits = limits or DEFAULT_LIMITS
    if check_models:
        expected_paths = list(
            itertools.chain.from_iterable(
                [
                    [
                        (
                            os.path.join(
                                models_directory,
                                experiment,
                                model,
                                f"{generate_experiment_name(experiment, model, seed, iterations, batch_size, limit)}.pt",
                            )
                        )
                        for seed in range(10)
                    ]
                    for limit in DEFAULT_LIMITS
                ]
            )
        )
        paths_exist = dict({p: os.path.exists(p) for p in expected_paths})

        for p, e in paths_exist.items():
            if not e:
                print(f"Expected {p} to exist")

        if not all(paths_exist.values()):
            raise RuntimeError("Data incomplete!")

    dataframes = [
        [
            dataframe_for_experiment(
                experiment,
                model,
                generate_experiment_name(
                    experiment, model, seed, iterations, batch_size, limit
                ),
                models_directory=models_directory,
                logs_directory=logs_directory,
                events=[keep_top_n_event, select_event],
                check_model=check_models,
            )
            for seed in trange(10)
        ]
        for limit in tqdm(limits)
    ]

    for seed_dfs, limit in zip(dataframes, limits):
        for df, seed in zip(seed_dfs, range(10)):
            if df["step"].max() < 0.9 * cut_at:
                print(
                    f"{generate_experiment_name(experiment, model, seed, iterations, batch_size, limit)} has only {df['step'].max()}"
                )

    return np.stack(
        [
            np.stack(
                [
                    keep_top(drop_after(df, "step", cut_at), keep_top_n_event, 10)[
                        select_event
                    ].values
                    for df in seed_df
                ]
            )
            for seed_df in dataframes
        ]
    )


def generate_sample_efficiency_matrix_mp(
    experiment,
    model,
    iterations,
    batch_size,
    keep_top_n_event,
    select_event,
    cut_at,
    models_directory,
    logs_directory,
    limits=None,
    check_models=True,
    n_procs=None,
):
    if check_models:
        check_models_exist(experiment, model, models_directory, limits)

    with multiprocessing.Pool(processes=n_procs) as pool:
        configurations = list(sample_efficiency_matrix_configurations(limits))
        sample_efficiency_configs_and_matrices = tqdm(
            pool.imap_unordered(
                functools.partial(
                    generate_sample_efficiency_matrix_from_config,
                    experiment,
                    model,
                    iterations,
                    batch_size,
                    keep_top_n_event,
                    select_event,
                    cut_at,
                    models_directory,
                    logs_directory,
                ),
                configurations,
            ),
            total=len(configurations),
        )

        sample_efficiency_configs_and_matrices = sorted(
            sample_efficiency_configs_and_matrices, key=lambda x: (x[0], x[1])
        )

        for warning in map(
            operator.itemgetter(-1), sample_efficiency_configs_and_matrices
        ):
            if warning:
                print(warning)

        sample_efficiency_configs_and_matrices = [
            c[:-1] for c in sample_efficiency_configs_and_matrices
        ]

        arrays = [
            [array for seed, limit, array in g]
            for k, g in itertools.groupby(
                sample_efficiency_configs_and_matrices, key=lambda x: x[0]
            )
        ]

        return np.stack([np.stack(seed_arrays) for seed_arrays in arrays])


def process_experiment_names_from_args(experiment_names):
    return [name.split(":", maxsplit=3) for name in experiment_names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs_dir", type=str)
    parser.add_argument("models_dir", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--limits", type=int, nargs="*")
    parser.add_argument(
        "--check-experiments",
        type=str,
        nargs="*",
        help="Experiments in the format experiment:model:iterations:cut_at",
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Whether to check if the models exist",
    )
    parser.add_argument("--n-procs", type=int, default=4)
    args = parser.parse_args()

    experiments_and_models = process_experiment_names_from_args(
        args.check_experiments or DEFAULT_EXPERIMENTS
    )
    id_sample_efficiency_matrices = [
        (
            experiment,
            model,
            generate_sample_efficiency_matrix_mp(
                experiment,
                model,
                int(n_iterations),
                32,
                "vsucc/dataloader_idx_0",
                "vsucc/dataloader_idx_0",
                int(cut_at),
                models_directory=args.models_dir,
                logs_directory=args.logs_dir,
                limits=args.limits or DEFAULT_LIMITS,
                check_models=args.check_models,
                n_procs=args.n_procs,
            ),
        )
        for experiment, model, n_iterations, cut_at in tqdm(
            experiments_and_models, desc="in-distribution data"
        )
    ]

    ood_sample_efficiency_matrices = [
        (
            experiment,
            model,
            generate_sample_efficiency_matrix_mp(
                experiment,
                model,
                int(n_iterations),
                32,
                "vsucc/dataloader_idx_0",
                "vsucc/dataloader_idx_1",
                int(cut_at),
                models_directory=args.models_dir,
                logs_directory=args.logs_dir,
                limits=args.limits or DEFAULT_LIMITS,
                check_models=args.check_models,
                n_procs=args.n_procs,
            ),
        )
        for experiment, model, n_iterations, cut_at in tqdm(
            experiments_and_models, desc="out-of-distribution data"
        )
    ]

    id_plot_data_means, id_plot_data_cis = to_lists(
        generate_sample_efficiency_curves_plot_data(
            {
                f"{experiment}:{model}": df
                for experiment, model, df in id_sample_efficiency_matrices
            }
        )
    )
    ood_plot_data_means, ood_plot_data_cis = to_lists(
        generate_sample_efficiency_curves_plot_data(
            {
                f"{experiment}:{model}": df
                for experiment, model, df in ood_sample_efficiency_matrices
            }
        )
    )

    with open(args.output_file, "w") as f:
        f.write(
            json.dumps(
                {
                    "id": {"means": id_plot_data_means, "cis": id_plot_data_cis},
                    "ood": {"means": ood_plot_data_means, "cis": ood_plot_data_cis},
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
