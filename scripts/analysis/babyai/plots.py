import fnmatch
import operator
import os
import itertools
import pickle
import babyaiutil

from babyaiutil.models.discriminator.film import FiLMConvEncoder
from babyaiutil.models.discriminator.transformer import TransformerEncoderDecoderModel
from babyaiutil.models.discriminator.independent_attention import (
    IndependentAttentionModel,
)
from babyaiutil.models.discriminator.simple_attention import SimpleAttentionModel
from babyaiutil.render import render_integer_encoded_grid

import tqdm.auto as tqdm
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import yaml
from rliable import library as rly
from rliable import metrics as rl_metrics
from rliable import plot_utils as rl_plot_utils
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

with open("./padded_bot_trajectories_10000_each.pb", "rb") as f:
    (train_trajectories, valid_trajectories, words, word2idx) = pickle.load(f)


SCRIPTS_DIR = os.path.abspath(
    os.path.join(
        os.path.expanduser("~"), "triton", "babyai-research", "babyaiutil", "scripts"
    ),
)


SCRIPTS_LOGS_DIR = os.path.join(SCRIPTS_DIR, "logs", "models")

SCRIPTS_MODELS_DIR = os.path.join(SCRIPTS_DIR, "models")


def get_most_recent_version(experiment_dir):
    versions = os.listdir(os.path.join(experiment_dir, "lightning_logs"))
    return sorted(versions, key=lambda x: int(x.split("_")[1]))[-1]


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


def generate_experiment_name(exp, model, seed, iterations, batch_size, limit):
    return f"{exp}_s_{seed}_m_{model}_it_{iterations}_b_{batch_size}_l_{limit}"


def long_to_wide(df):
    return df.pivot_table(
        index=["event"], columns="step", values="value"
    ).T.reset_index()


def dataframe_for_experiment(
    task_name, model_name, experiment_name, events=None, check_model=True
):
    model_path = os.path.join(
        SCRIPTS_MODELS_DIR, task_name, model_name, f"{experiment_name}.pt"
    )
    if check_model and not os.path.exists(model_path):
        raise RuntimeError(f"{model_path} doesn't exist, experiment may be incomplete")

    df = tb_events_scalars_to_pd_dataframe(
        EventAccumulator(
            get_tb_logs(SCRIPTS_LOGS_DIR, task_name, model_name, experiment_name)
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
                    generate_experiment_name(exp, model, seed, iterations, limit),
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
    limits=[50, 100, 250, 500, 1000, 2500, 5000, 9980],
    events=["dataloader_idx_0/vsucc", "dataloader_idx_1/vsucc"],
):
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


limits = [50, 100, 250, 500, 1000, 2500, 5000, 9980]


def generate_sample_efficiency_matrix(
    experiment,
    model,
    iterations,
    batch_size,
    keep_top_n_event,
    select_event,
    cut_at,
    check_models=True,
):
    if check_models:
        expected_paths = list(
            itertools.chain.from_iterable(
                [
                    [
                        (
                            os.path.join(
                                SCRIPTS_MODELS_DIR,
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

    return np.stack(
        [
            np.stack(
                [
                    keep_top(
                        drop_after(
                            dataframe_for_experiment(
                                experiment,
                                model,
                                generate_experiment_name(
                                    experiment,
                                    model,
                                    seed,
                                    iterations,
                                    batch_size,
                                    limit,
                                ),
                                events=[keep_top_n_event, select_event],
                                check_model=check_models,
                            ),
                            "step",
                            cut_at,
                        ),
                        keep_top_n_event,
                        10,
                    )[select_event].values
                    for seed in tqdm.trange(10)
                ]
            )
            for limit in tqdm.tqdm(limits)
        ]
    )


vin_mvprop_df_id = generate_sample_efficiency_matrix(
    "vin_sample_nogoal_16_logsumexp_ignorance",
    "mvprop",
    70000,
    32,
    "vsucc/dataloader_idx_0",
    "vsucc/dataloader_idx_0",
    70000,
)


vin_mvprop_df_ood = generate_sample_efficiency_matrix(
    "vin_sample_nogoal_16_logsumexp_ignorance",
    "mvprop",
    70000,
    32,
    "vsucc/dataloader_idx_0",
    "vsucc/dataloader_idx_1",
    70000,
)


vin_noprop_df_id = generate_sample_efficiency_matrix(
    "vin_sample_nogoal_16_logsumexp_k_0",
    "mvprop",
    70000,
    32,
    "vsucc/dataloader_idx_0",
    "vsucc/dataloader_idx_0",
    70000,
)


vin_noprop_df_ood = generate_sample_efficiency_matrix(
    "vin_sample_nogoal_16_logsumexp_k_0",
    "mvprop",
    70000,
    32,
    "vsucc/dataloader_idx_0",
    "vsucc/dataloader_idx_1",
    70000,
)


imitation_pure_transformer_df_id = generate_sample_efficiency_matrix(
    "imitation",
    "pure_transformer",
    120000,
    32,
    "vsucc/dataloader_idx_0",
    "vsucc/dataloader_idx_0",
    70000,
    check_models=False,
)


imitation_pure_transformer_df_ood = generate_sample_efficiency_matrix(
    "imitation",
    "pure_transformer",
    120000,
    32,
    "vsucc/dataloader_idx_0",
    "vsucc/dataloader_idx_1",
    70000,
    check_models=False,
)

imitation_fused_inputs_next_step_encoder_dropout_df_id = (
    generate_sample_efficiency_matrix(
        "imitation",
        "fused_inputs_next_step_encoder",
        250000,
        32,
        "vsucc/dataloader_idx_0",
        "vsucc/dataloader_idx_0",
        250000,
        check_models=False,
    )
)


imitation_fused_inputs_next_step_encoder_dropout_df_ood = (
    generate_sample_efficiency_matrix(
        "imitation",
        "fused_inputs_next_step_encoder",
        250000,
        32,
        "vsucc/dataloader_idx_0",
        "vsucc/dataloader_idx_1",
        250000,
        check_models=False,
    )
)


imitation_film_lstm_df_id = generate_sample_efficiency_matrix(
    "imitation",
    "film_lstm_policy",
    70000,
    32,
    "vsucc/dataloader_idx_0",
    "vsucc/dataloader_idx_0",
    70000,
    check_models=False,
)


imitation_film_lstm_df_ood = generate_sample_efficiency_matrix(
    "imitation",
    "film_lstm_policy",
    70000,
    32,
    "vsucc/dataloader_idx_0",
    "vsucc/dataloader_idx_1",
    70000,
    check_models=False,
)


# In[28]:


def aggregate_func(x):
    return np.array([rl_metrics.aggregate_iqm(x[..., i]) for i in range(x.shape[-1])])


# In[29]:


def sample_efficiency_curves(experiment_dfs, ylabel=None, xlabel=None):
    aggregate_iqms, aggregate_cis = rly.get_interval_estimates(
        {k: df.transpose(1, 2, 0) for k, df in experiment_dfs.items()}, aggregate_func
    )

    ax = rl_plot_utils.plot_sample_efficiency_curve(
        limits,
        aggregate_iqms,
        aggregate_cis,
        algorithms=None,
        xlabel="Samples per goal",
        ylabel=ylabel or "Success rate",
    )
    ax.set_xscale("log")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1))
    ax.set_ylim((-0.05, 1.05))
    ax.spines["left"].set_position(("outward", 0))
    ax.spines["bottom"].set_position(("outward", 0))

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(12)

    return ax, aggregate_iqms, aggregate_cis


plt.rcParams["text.usetex"] = True

os.makedirs("figures", exist_ok=True)
ax, imitation_ood_iqms, imitation_ood_cis = sample_efficiency_curves(
    {
        "GRU-Encoder ResNet/FiLM Decoder": imitation_film_lstm_df_ood,
        "Encoder-Decoder Transformer": imitation_pure_transformer_df_ood,
        # "Encoder-only Transformer": imitation_fused_inputs_next_step_encoder_dropout_df_ood,
        "Compositional/CNN Policy": vin_noprop_df_ood,
        "Compositional/MVProp Planner (ours)": vin_mvprop_df_ood,
    }
)

ax.set_title("$\\mathcal{D}_{\\textrm{v\\_OOD}}$\n")

for i in range(3):
    ax.text(
        limits[-i - 3],
        imitation_ood_iqms["Compositional/MVProp Planner (ours)"][-i - 3] + 0.03,
        f"{imitation_ood_iqms['Compositional/MVProp Planner (ours)'][-i - 3]:.2}"[1:],
    )
    ax.text(
        limits[i],
        imitation_ood_iqms["Compositional/MVProp Planner (ours)"][i] + 0.03,
        f"{imitation_ood_iqms['Compositional/MVProp Planner (ours)'][i]:.2}"[1:],
    )

    ax.text(
        limits[-i - 1],
        imitation_ood_iqms["GRU-Encoder ResNet/FiLM Decoder"][-i - 1] + 0.03,
        f"{imitation_ood_iqms['GRU-Encoder ResNet/FiLM Decoder'][-i - 1]:.2}"[1:],
    )

    ax.text(
        limits[-i - 1],
        imitation_ood_iqms["Compositional/CNN Policy"][-i - 1] + 0.03,
        f"{imitation_ood_iqms['Compositional/CNN Policy'][-i - 1]:.2}"[1:],
    )

    ax.text(
        limits[-i - 1],
        imitation_ood_iqms["Encoder-Decoder Transformer"][-i - 1] - 0.05,
        f"{imitation_ood_iqms['Encoder-Decoder Transformer'][-i - 1]:.2}"[1:],
    )


# In[131]:


ax.get_figure().savefig("figures/imitation_ood_success_rates.pdf")


# In[132]:


ax, imitation_id_iqms, imitation_id_cis = sample_efficiency_curves(
    {
        "GRU-Encoder ResNet/FiLM Decoder": imitation_film_lstm_df_id,
        "Encoder-Decoder Transformer": imitation_pure_transformer_df_id,
        # "Encoder-only Transformer": imitation_fused_inputs_next_step_encoder_dropout_df_id,
        "Compositional/CNN Policy": vin_noprop_df_id,
        "Compositional/MVProp Planner (ours)": vin_mvprop_df_id,
    }
)

ax.set_title("$\\mathcal{D}_{\\textrm{v\\_ID}}$\n")

for i in range(3):
    ax.text(
        limits[i],
        imitation_id_iqms["Compositional/MVProp Planner (ours)"][i] + 0.03,
        f"{imitation_id_iqms['Compositional/MVProp Planner (ours)'][i]:.2}"[1:],
    )

    ax.text(
        limits[-i - 4],
        imitation_id_iqms["Encoder-Decoder Transformer"][-i - 4] - 0.03,
        f"{imitation_id_iqms['Encoder-Decoder Transformer'][-i - 4]:.2}"[1:],
    )

ax.get_figure().savefig("figures/imitation_id_success_rates.pdf")

# Discriminator models


discriminator_simple_attention_df_id = generate_sample_efficiency_matrix(
    "discriminator_sample_nogoal_16_logsumexp",
    "simple",
    200000,
    1024,
    "vsf1/dataloader_idx_0",
    "vsf1/dataloader_idx_0",
    200000,
)


discriminator_simple_attention_df_ood = generate_sample_efficiency_matrix(
    "discriminator_sample_nogoal_16_logsumexp",
    "simple",
    200000,
    1024,
    "vsf1/dataloader_idx_0",
    "vsf1/dataloader_idx_1",
    200000,
)


discriminator_independent_df_id = generate_sample_efficiency_matrix(
    "discriminator_sample_nogoal_16_logsumexp",
    "independent",
    200000,
    1024,
    "vsf1/dataloader_idx_0",
    "vsf1/dataloader_idx_0",
    200000,
)

discriminator_independent_df_ood = generate_sample_efficiency_matrix(
    "discriminator_sample_nogoal_16_logsumexp",
    "independent",
    200000,
    1024,
    "vsf1/dataloader_idx_0",
    "vsf1/dataloader_idx_1",
    200000,
)


discriminator_independent_noreg_df_id = generate_sample_efficiency_matrix(
    "discriminator_sample_nogoal_16_logsumexp",
    "independent_noreg",
    200000,
    1024,
    "vsf1/dataloader_idx_0",
    "vsf1/dataloader_idx_0",
    200000,
)


discriminator_independent_noreg_df_ood = generate_sample_efficiency_matrix(
    "discriminator_sample_nogoal_16_logsumexp",
    "independent_noreg",
    200000,
    1024,
    "vsf1/dataloader_idx_0",
    "vsf1/dataloader_idx_1",
    200000,
)

discriminator_transformer_df_id = generate_sample_efficiency_matrix(
    "discriminator_sample_nogoal_16_logsumexp",
    "transformer",
    200000,
    1024,
    "vsf1/dataloader_idx_0",
    "vsf1/dataloader_idx_0",
    200000,
)


discriminator_transformer_df_ood = generate_sample_efficiency_matrix(
    "discriminator_sample_nogoal_16_logsumexp",
    "transformer",
    200000,
    1024,
    "vsf1/dataloader_idx_0",
    "vsf1/dataloader_idx_1",
    200000,
)


discriminator_film_df_id = generate_sample_efficiency_matrix(
    "discriminator_sample_nogoal_16_logsumexp",
    "film",
    200000,
    1024,
    "vsf1/dataloader_idx_0",
    "vsf1/dataloader_idx_0",
    200000,
)


discriminator_film_df_ood = generate_sample_efficiency_matrix(
    "discriminator_sample_nogoal_16_logsumexp",
    "film",
    200000,
    1024,
    "vsf1/dataloader_idx_0",
    "vsf1/dataloader_idx_1",
    200000,
)


ax, discriminator_id_iqms, discriminator_id_cis = sample_efficiency_curves(
    {
        "FiLM": discriminator_film_df_id,
        "Transformer": discriminator_transformer_df_id,
        "Independent, no L1 Reg": discriminator_independent_noreg_df_id,
        "Independent": discriminator_independent_df_id,
    },
    ylabel="Soft F1 Score",
)
ax.get_figure().savefig("figures/discriminator_f1_curves_id.pdf")


(
    aggregate_iqms_discrminator_id,
    aggregate_cis_discriminator_id,
) = rly.get_interval_estimates(
    {
        k: df.transpose(1, 2, 0).reshape(df.shape[1], -1, 1)
        for k, df in {
            "FiLM": discriminator_film_df_id,
            "Transformer": discriminator_transformer_df_id,
            "Attention, L1": discriminator_simple_attention_df_id,
            "Independent": discriminator_independent_noreg_df_id,
            "Independent, L1": discriminator_independent_df_id,
        }.items()
    },
    aggregate_func,
)


# In[117]:


(
    aggregate_iqms_discrminator_ood,
    aggregate_cis_discriminator_ood,
) = rly.get_interval_estimates(
    {
        k: df.transpose(1, 2, 0).reshape(df.shape[1], -1, 1)
        for k, df in {
            "FiLM": discriminator_film_df_ood,
            "Transformer": discriminator_transformer_df_ood,
            "Attention, L1": discriminator_simple_attention_df_ood,
            "Independent": discriminator_independent_noreg_df_ood,
            "Independent, L1": discriminator_independent_df_ood,
        }.items()
    },
    aggregate_func,
)


aggregate_discriminator_stats_table = pd.DataFrame.from_dict(
    {
        k: [
            f"{aggregate_iqms_discrminator_id[k][0]:.3f} ± {aggregate_cis_discriminator_id[k][-1][0] - aggregate_cis_discriminator_id[k][0][0]:.3f}",
            f"{aggregate_iqms_discrminator_ood[k][0]:.3f} ± {aggregate_cis_discriminator_ood[k][-1][0] - aggregate_cis_discriminator_ood[k][0][0]:.3f}",
        ]
        for k in aggregate_iqms_discrminator_id
    }
)
aggregate_discriminator_stats_table.index = ["In-Distribution", "Out of Distribution"]
print(aggregate_discriminator_stats_table.T.to_latex())

ax, discriminator_ood_iqms, discriminator_ood_cis = sample_efficiency_curves(
    {
        "FiLM": discriminator_film_df_ood,
        "Transformer": discriminator_transformer_df_ood,
        "Attention": discriminator_simple_attention_df_ood,
        "Independent, no L1 Reg": discriminator_independent_noreg_df_ood,
        "Independent": discriminator_independent_df_ood,
    },
    ylabel="Soft F1 Score",
)
ax.get_figure().savefig("figures/discriminator_f1_curves_ood.pdf")


# # Visualize Embedding Dot Products on the best performing discriminator models


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


discriminator_task_limits = [10, 50, 100, 250, 500, 1000, 2500, 5000, 9980]
words = sorted(
    "go to the a red blue purple green grey yellow box key ball door [act]".split()
)


def get_checkpoint_scores_tuples_for_models(
    task_name, model_name, iterations, batch_size, seeds, limits
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
                                SCRIPTS_LOGS_DIR,
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


independent_scores = get_checkpoint_scores_tuples_for_models(
    "discriminator_sample_nogoal_16_logsumexp",
    "independent",
    200000,
    1024,
    range(10),
    discriminator_task_limits,
)


independent_deltas = compute_deltas(independent_scores)


independent_noreg_scores = get_checkpoint_scores_tuples_for_models(
    "discriminator_sample_nogoal_16_logsumexp",
    "independent_noreg",
    200000,
    1024,
    range(10),
    limits,
)


independent_noreg_deltas = compute_deltas(independent_noreg_scores)


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


independent_row_indices = get_best_rows_indices(independent_deltas)


independent_noreg_row_indices = get_best_rows_indices(independent_noreg_deltas)


def trim_rows_indices_pcts(rows, limit_lowerbound=10, limit_upperbound=500, pct=0.3):
    return [
        (limit, trim_row_range_percent(row, pct))
        for limit, row in rows
        if limit >= limit_lowerbound and limit <= limit_upperbound
    ]


independent_row_indices = trim_rows_indices_pcts(
    independent_scores, limit_lowerbound=10, limit_upperbound=1000, pct=0.75
)
independent_row_indices

independent_noreg_row_indices = trim_rows_indices_pcts(
    independent_noreg_scores, limit_lowerbound=10, limit_upperbound=1000, pct=0.75
)
independent_noreg_row_indices


def normalize(vec):
    return vec / np.linalg.norm(vec, axis=-1)[:, None]


def compute_normalized_outer_product(first, second):
    result = normalize(first) @ normalize(second).T
    return result


def fetch_numpy_array_from_sd(state_dict, key):
    return state_dict[key].cpu().numpy()


def flatten_tree(tree):
    for limit, seeds in tree:
        for seed in seeds:
            yield (seed, int(limit))


def seed_limit_pairs_to_model_paths(
    task_name, model_name, iterations, batch_size, seed_limit_pairs
):
    return [
        os.path.join(
            SCRIPTS_DIR,
            checkpoint_score_tuple[0],
        )
        for checkpoint_score_tuple in [
            get_path_to_best_scoring_model(
                get_best_k_models(
                    SCRIPTS_LOGS_DIR,
                    task_name,
                    model_name,
                    generate_experiment_name(
                        task_name, model_name, seed, iterations, batch_size, limit
                    ),
                )
            )
            for seed, limit in seed_limit_pairs
        ]
        if checkpoint_score_tuple is not None
    ]


def compute_affine_sigmoid_transform_from_sd(array, sd):
    logits = (
        sd["model.projection.weight"].exp().numpy() * np.clip(array, 0, 1) * 2
        + sd["model.projection.bias"].numpy()
    )

    clip_logits = np.clip(logits, -8, 8)
    result = np.exp(clip_logits) / (1 + np.exp(clip_logits))

    return result


def get_embedding_correlation_matrices_for_models(model_paths):
    return np.stack(
        [
            compute_affine_sigmoid_transform_from_sd(
                compute_normalized_outer_product(
                    fetch_numpy_array_from_sd(sd, "model.word_embeddings.weight"),
                    fetch_numpy_array_from_sd(sd, "model.attrib_embeddings.weight"),
                ),
                sd,
            )
            for sd in map(
                lambda x: torch.load(x, map_location=torch.device("cpu"))["state_dict"],
                model_paths,
            )
        ]
    )


independent_noreg_model_paths = seed_limit_pairs_to_model_paths(
    "discriminator_sample_nogoal_16_logsumexp",
    "independent_noreg",
    200000,
    1024,
    flatten_tree(independent_noreg_row_indices),
)


independent_model_paths = seed_limit_pairs_to_model_paths(
    "discriminator_sample_nogoal_16_logsumexp",
    "independent",
    200000,
    1024,
    flatten_tree(independent_noreg_row_indices),
)

independent_noreg_correlation_matrices = get_embedding_correlation_matrices_for_models(
    independent_noreg_model_paths
)


independent_correlation_matrices = get_embedding_correlation_matrices_for_models(
    independent_model_paths
)


simple_attention_scores = get_checkpoint_scores_tuples_for_models(
    "discriminator_sample_nogoal_16_logsumexp",
    "simple",
    200000,
    1024,
    range(10),
    discriminator_task_limits,
)


simple_attention_indices = trim_rows_indices_pcts(
    simple_attention_scores, limit_lowerbound=10, limit_upperbound=1000, pct=0.75
)
simple_attention_indices

simple_attention_model_paths = seed_limit_pairs_to_model_paths(
    "discriminator_sample_nogoal_16_logsumexp",
    "simple",
    200000,
    1024,
    flatten_tree(simple_attention_indices),
)


def compute_projected_pairs(sd, embeddings_weight):
    embed_weight = embeddings_weight[sd["attrib_ranges_expanded"].numpy()].reshape(
        66, 2 * 48
    )
    projected = (
        embed_weight @ sd["model.in_projection.weight"].numpy().T
        + sd["model.in_projection.bias"].numpy()
    )

    return projected


def get_projected_embedding_correlation_matrices_for_models(model_paths):
    return np.stack(
        [
            compute_affine_sigmoid_transform_from_sd(
                compute_normalized_outer_product(
                    fetch_numpy_array_from_sd(sd, "model.word_embeddings.weight"),
                    compute_projected_pairs(
                        sd,
                        fetch_numpy_array_from_sd(sd, "model.attrib_embeddings.weight"),
                    ),
                ),
                sd,
            )
            for sd in map(
                lambda x: torch.load(x, map_location=torch.device("cpu"))["state_dict"],
                model_paths,
            )
        ]
    )


simple_correlation_matrices = get_projected_embedding_correlation_matrices_for_models(
    simple_attention_model_paths
)


def plt_normalized_heatmap_with_numbers(ax, matrix, cmap=None, threshold=0.01, vmax=1):
    table_data = np.round(matrix, decimals=2)

    ax.imshow(table_data, cmap=cmap or "viridis", vmin=0, vmax=vmax, aspect="auto")

    for i in range(table_data.shape[0]):
        for j in range(table_data.shape[1]):
            if np.abs(table_data[i, j]) > threshold:
                text = ax.text(
                    j,
                    i,
                    str(table_data[i, j])[:4],
                    ha="center",
                    va="center",
                    color="w" if (table_data[i, j] < 0.8) else "blue",
                )


def plot_embedding_correlation_matrix_to_ax(ax, matrix, cmap=None, threshold=0.01):
    plt_normalized_heatmap_with_numbers(ax, matrix, cmap=cmap, threshold=threshold)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        list(OBJECT_TO_IDX.keys()) + list(COLOR_TO_IDX.keys()), rotation=40, fontsize=16
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(words, fontsize=16)


def plot_embedding_correlation_matrix(
    matrix, cmap=None, save_filename=None, threshold=0.01
):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_embedding_correlation_matrix_to_ax(ax, matrix, cmap=cmap, threshold=threshold)

    if save_filename:
        fig.savefig(save_filename)


plot_embedding_correlation_matrix(
    independent_correlation_matrices.mean(axis=0),
    save_filename="independent_mean_embeddings_correlation.pdf",
)


plot_embedding_correlation_matrix(
    independent_correlation_matrices.std(axis=0),
    save_filename="independent_std_embeddings_correlation.pdf",
    cmap="inferno",
)


plot_embedding_correlation_matrix(
    independent_noreg_correlation_matrices.mean(axis=0),
    save_filename="independent_noreg_mean_embeddings_correlation.pdf",
)

plot_embedding_correlation_matrix(
    independent_noreg_correlation_matrices.std(axis=0),
    save_filename="independent_noreg_std_embeddings_correlation.pdf",
    cmap="inferno",
)


def plot_projected_embedding_correlation_matrix_to_ax(
    ax, matrix, cmap=None, threshold=None
):
    # Need to figure out to label these
    plt_normalized_heatmap_with_numbers(
        ax, matrix.T[5 * 6 : 8 * 6].T, cmap=cmap, threshold=threshold
    )
    ax.set_yticks(np.arange(len(words)))
    ax.set_yticklabels(words, fontsize=16)
    ax.set_xticklabels(
        [
            " ".join(
                [
                    list(COLOR_TO_IDX.keys())[j - len(OBJECT_TO_IDX)],
                    list(OBJECT_TO_IDX.keys())[i],
                ]
            )
            for i, j in list(
                itertools.product(
                    range(0, len(OBJECT_TO_IDX.keys())),
                    range(
                        len(OBJECT_TO_IDX.keys()),
                        len(COLOR_TO_IDX.keys()) + len(OBJECT_TO_IDX.keys()),
                    ),
                )
            )[5 * 6 : 8 * 6]
        ],
        rotation=40,
        ha="right",
        fontsize=16,
    )
    ax.set_xticks(range(3 * 6))


def plot_projected_embedding_correlation_matrix(
    matrix, save_filename=None, cmap=None, threshold=None
):
    # Need to figure out to label these
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_projected_embedding_correlation_matrix_to_ax(
        ax, matrix, cmap=cmap, threshold=threshold
    )

    if save_filename:
        fig.savefig(save_filename)


plot_projected_embedding_correlation_matrix(
    simple_correlation_matrices.mean(axis=0),
    save_filename="simple_projected_embeddings_correlation.pdf",
    threshold=0.03,
)


def plot_correlations_2x2(
    independent_noreg_mean_correlations,
    independent_noreg_std_correlations,
    independent_correlations,
    simple_mean_correlations,
    threshold=0.03,
    savefig=None,
    **rc_params,
):
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    plot_embedding_correlation_matrix_to_ax(
        ax[0][0],
        independent_noreg_mean_correlations,
        cmap="viridis",
        threshold=threshold,
    )
    ax[0][0].set_ylabel("word")
    ax[0][0].set_xlabel("Independent Attention mean object/color corr")

    plot_embedding_correlation_matrix_to_ax(
        ax[0][1],
        independent_noreg_std_correlations,
        cmap="inferno",
        threshold=threshold,
    )
    ax[0][1].set_yticklabels([])
    ax[0][1].get_yaxis().set_visible(False)
    ax[0][1].set_xlabel("Independent Attention  std. of object/color corr")

    plot_projected_embedding_correlation_matrix_to_ax(
        ax[1][0], simple_mean_correlations, cmap="viridis", threshold=threshold
    )
    ax[1][0].set_ylabel("word")
    ax[1][0].set_xlabel("Sparse Attention mean object/color corr")

    plot_embedding_correlation_matrix_to_ax(
        ax[1][1], independent_correlations, cmap="viridis", threshold=threshold
    )
    ax[1][1].set_yticklabels([])
    ax[1][1].get_yaxis().set_visible(False)
    ax[1][1].set_xlabel("Sparse Compositional Attention mean object/color corr")

    for a in ax.ravel():
        a.set_anchor("N")
        for item in [a.title, a.xaxis.label, a.yaxis.label]:
            item.set_fontsize(15)

    fig.subplots_adjust(wspace=0.025, hspace=0.1)

    if savefig:
        fig.savefig(savefig)


def plot_correlations_3x1(
    simple_mean_correlations,
    independent_noreg_mean_correlations,
    independent_correlations,
    threshold=0.03,
    savefig=None,
    **rc_params,
):
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))

    plot_projected_embedding_correlation_matrix_to_ax(
        ax[0], simple_mean_correlations, cmap="viridis", threshold=threshold
    )
    ax[0].get_yaxis().set_visible(True)
    ax[0].set_ylabel("word")
    ax[0].set_xlabel("Sparse Attention")

    plot_embedding_correlation_matrix_to_ax(
        ax[1], independent_noreg_mean_correlations, cmap="viridis", threshold=threshold
    )
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_ylabel("word")
    ax[1].set_xlabel("Compositional Attention")

    plot_embedding_correlation_matrix_to_ax(
        ax[2], independent_correlations, cmap="viridis", threshold=threshold
    )
    ax[2].get_yaxis().set_visible(False)
    ax[2].set_xlabel("Sparse Compositional Attention")

    for a in ax.ravel():
        a.set_anchor("N")
        for item in [a.title, a.xaxis.label, a.yaxis.label]:
            item.set_fontsize(28)

    fig.subplots_adjust(wspace=0.025, hspace=0.1)
    fig.align_xlabels()

    if savefig:
        fig.savefig(savefig, bbox_inches="tight")


plot_correlations_3x1(
    simple_correlation_matrices.mean(axis=0),
    independent_noreg_correlation_matrices.mean(axis=0),
    independent_correlation_matrices.mean(axis=0),
    savefig="independent_correlation_heatmaps_3x1.pdf",
)


# # Evaluating qualitative model performance on training/test
def translate_state_dict(sd, strip="encoder."):
    return {k[len(strip) :]: v for k, v in sd.items() if k.startswith(strip)}


def predict_from_model_path(
    model_cls, state_dict_key_prefix, model_path, input_frame, input_mission
):
    model = model_cls(
        [0, len(OBJECT_TO_IDX), len(OBJECT_TO_IDX) + len(COLOR_TO_IDX)],
        48,
        len(words),
    )
    model.load_state_dict(
        translate_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))["state_dict"],
            state_dict_key_prefix,
        )
    )

    return (
        model(
            torch.from_numpy(input_frame)[..., :2].long(),
            torch.from_numpy(input_mission).long(),
        )[0][..., 0]
        .sigmoid()
        .detach()
        .numpy()
    )


def predict_evaluations(
    model_cls, state_dict_key_prefix, model_paths, input_frame, input_mission
):
    return np.stack(
        [
            predict_from_model_path(
                model_cls, state_dict_key_prefix, model_path, input_frame, input_mission
            )
            for model_path in model_paths
        ]
    )


transformer_scores = get_checkpoint_scores_tuples_for_models(
    "discriminator_sample_nogoal_16",
    "transformer",
    100000,
    1024,
    range(10),
    discriminator_task_limits,
)

transformer_row_indices = trim_rows_indices_pcts(
    transformer_scores, limit_lowerbound=10, limit_upperbound=1000, pct=0.75
)
transformer_row_indices

transformer_model_paths = seed_limit_pairs_to_model_paths(
    "discriminator_sample_nogoal_16_logsumexp",
    "transformer",
    200000,
    1024,
    flatten_tree(transformer_row_indices),
)

transformer_model_training_evaluations = predict_evaluations(
    babyaiutil.models.discriminator.transformer.TransformerEncoderDecoderModel,
    "encoder.",
    transformer_model_paths,
    train_trajectories["image_trajectories"][1][0][None],
    train_trajectories["missions"][1][None],
)[:, 0]

film_scores = get_checkpoint_scores_tuples_for_models(
    "discriminator_sample_nogoal_16_logsumexp",
    "film",
    200000,
    1024,
    range(10),
    discriminator_task_limits,
)

film_row_indices = trim_rows_indices_pcts(
    film_scores, limit_lowerbound=10, limit_upperbound=1000, pct=0.75
)
film_row_indices


film_model_paths = seed_limit_pairs_to_model_paths(
    "discriminator_sample_nogoal_16_logsumexp",
    "film",
    200000,
    1024,
    flatten_tree(film_row_indices),
)


def plot_qualitative(
    rendered_training_input,
    rendered_training_label,
    rendered_validation_input,
    rendered_validation_label,
    film_training,
    film_validation,
    film_validation_std,
    transformer_training,
    transformer_validation,
    transformer_validation_std,
    independent_noreg_training,
    independent_noreg_validation,
    independent_noreg_validation_std,
    simple_attention_training,
    simple_attention_validation,
    simple_attention_validation_std,
    independent_training,
    independent_validation,
    independent_validation_std,
    threshold=0.1,
):
    fig, ax = plt.subplots(3, 6, figsize=(22.5, 11.5))
    ax[0, 0].imshow(rendered_training_input)
    ax[0, 0].set_yticklabels([])
    ax[0, 0].set_xticklabels([])
    ax[0, 0].set_ylabel(
        "$\\mathcal{{D}}_{{\\textrm{{v\\_OOD}}}}$\n{}".format(rendered_training_label),
        multialignment="center",
    )
    ax[0, 1].imshow(film_training)
    plt_normalized_heatmap_with_numbers(ax[0, 1], film_training, threshold=threshold)
    ax[0, 1].axis("off")
    ax[0, 1].set_xticklabels([])
    ax[0, 1].set_yticklabels([])
    plt_normalized_heatmap_with_numbers(
        ax[0, 2], transformer_training, threshold=threshold
    )
    ax[0, 2].axis("off")
    ax[0, 2].set_xticklabels([])
    ax[0, 2].set_yticklabels([])
    plt_normalized_heatmap_with_numbers(
        ax[0, 3], independent_noreg_training, threshold=threshold
    )
    ax[0, 3].axis("off")
    ax[0, 3].set_xticklabels([])
    ax[0, 3].set_yticklabels([])
    plt_normalized_heatmap_with_numbers(
        ax[0, 4], simple_attention_training, threshold=threshold
    )
    ax[0, 4].axis("off")
    ax[0, 4].set_xticklabels([])
    ax[0, 4].set_yticklabels([])
    plt_normalized_heatmap_with_numbers(
        ax[0, 5], independent_training, threshold=threshold
    )
    ax[0, 5].axis("off")
    ax[0, 5].set_xticklabels([])
    ax[0, 5].set_yticklabels([])

    ax[1, 0].imshow(rendered_validation_input)
    ax[1, 0].set_yticklabels([])
    ax[1, 0].set_xticklabels([])
    ax[1, 0].set_ylabel(
        "$\\mathcal{{D}}_{{\\textrm{{v\\_OOD}}}}$\n{}".format(
            rendered_validation_label
        ),
        multialignment="center",
    )
    plt_normalized_heatmap_with_numbers(ax[1, 1], film_validation, threshold=threshold)
    ax[1, 1].set_yticklabels([])
    ax[1, 1].set_xticklabels([])
    ax[1, 1].axis("off")
    plt_normalized_heatmap_with_numbers(
        ax[1, 2], transformer_validation, threshold=threshold
    )
    ax[1, 2].set_yticklabels([])
    ax[1, 2].set_xticklabels([])
    ax[1, 2].axis("off")
    plt_normalized_heatmap_with_numbers(
        ax[1, 3], independent_noreg_validation, threshold=threshold
    )
    ax[1, 3].set_yticklabels([])
    ax[1, 3].set_xticklabels([])
    ax[1, 3].axis("off")
    plt_normalized_heatmap_with_numbers(
        ax[1, 4], simple_attention_validation, threshold=threshold
    )
    ax[1, 4].set_yticklabels([])
    ax[1, 4].set_xticklabels([])
    ax[1, 4].axis("off")
    plt_normalized_heatmap_with_numbers(
        ax[1, 5], independent_validation, threshold=threshold
    )
    ax[1, 5].set_yticklabels([])
    ax[1, 5].set_xticklabels([])
    ax[1, 5].axis("off")

    ax[2, 0].imshow(rendered_validation_input)
    ax[2, 0].set_yticklabels([])
    ax[2, 0].set_xticklabels([])
    ax[2, 0].set_ylabel(
        "$\\mathcal{{D}}_{{\\textrm{{v\\_OOD}}}}$ std.\n{}".format(
            rendered_validation_label
        ),
        multialignment="center",
    )
    plt_normalized_heatmap_with_numbers(
        ax[2, 1], film_validation_std, threshold=threshold, cmap="inferno"
    )
    ax[2, 1].set_yticklabels([])
    ax[2, 1].set_xticklabels([])
    ax[2, 1].set_xlabel("FiLM")
    plt_normalized_heatmap_with_numbers(
        ax[2, 2], transformer_validation_std, threshold=threshold, cmap="inferno"
    )
    ax[2, 2].set_yticklabels([])
    ax[2, 2].set_xticklabels([])
    ax[2, 2].set_xlabel("Transformer")
    plt_normalized_heatmap_with_numbers(
        ax[2, 3], independent_noreg_validation_std, threshold=threshold, cmap="inferno"
    )
    ax[2, 3].set_yticklabels([])
    ax[2, 3].set_xticklabels([])
    ax[2, 3].set_xlabel("Compositional Attention")
    plt_normalized_heatmap_with_numbers(
        ax[2, 4], simple_attention_validation_std, threshold=threshold, cmap="inferno"
    )
    ax[2, 4].set_yticklabels([])
    ax[2, 4].set_xticklabels([])
    ax[2, 4].set_xlabel("Sparse Attention")
    plt_normalized_heatmap_with_numbers(
        ax[2, 5], independent_validation_std, threshold=threshold, cmap="inferno"
    )
    ax[2, 5].set_yticklabels([])
    ax[2, 5].set_xticklabels([])
    ax[2, 5].set_xlabel("Sparse Compositional Attention")

    for a in ax.ravel():
        a.set_anchor("N")
        for item in [a.title, a.xaxis.label, a.yaxis.label]:
            item.set_fontsize(18)

    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    return fig


plt.rcParams["text.usetex"] = True


def evaluate_and_plot_qualitative(
    training_index, validation_index, savefig=None, threshold=0.1
):
    film_model_training_evaluations = predict_evaluations(
        babyaiutil.models.discriminator.film.FiLMConvEncoder,
        "film_encoder.",
        film_model_paths,
        train_trajectories["image_trajectories"][training_index][0][None],
        train_trajectories["missions"][training_index][None],
    )[:, 0]
    film_model_valid_evaluations = predict_evaluations(
        babyaiutil.models.discriminator.film.FiLMConvEncoder,
        "film_encoder.",
        film_model_paths,
        valid_trajectories["image_trajectories"][validation_index][0][None],
        valid_trajectories["missions"][validation_index][None],
    )[:, 0]

    transformer_model_training_evaluations = predict_evaluations(
        babyaiutil.models.discriminator.transformer.TransformerEncoderDecoderModel,
        "encoder.",
        transformer_model_paths,
        train_trajectories["image_trajectories"][training_index][0][None],
        train_trajectories["missions"][training_index][None],
    )[:, 0]

    transformer_model_valid_evaluations = predict_evaluations(
        babyaiutil.models.discriminator.transformer.TransformerEncoderDecoderModel,
        "encoder.",
        transformer_model_paths,
        valid_trajectories["image_trajectories"][validation_index][0][None],
        valid_trajectories["missions"][validation_index][None],
    )[:, 0]

    independent_noreg_model_training_evaluations = predict_evaluations(
        babyaiutil.models.discriminator.independent_attention.IndependentAttentionModel,
        "model.",
        independent_noreg_model_paths,
        train_trajectories["image_trajectories"][training_index][0],
        train_trajectories["missions"][training_index],
    )
    independent_noreg_model_valid_evaluations = predict_evaluations(
        babyaiutil.models.discriminator.independent_attention.IndependentAttentionModel,
        "model.",
        independent_noreg_model_paths,
        valid_trajectories["image_trajectories"][validation_index][0],
        valid_trajectories["missions"][validation_index],
    )

    independent_model_training_evaluations = predict_evaluations(
        babyaiutil.models.discriminator.independent_attention.IndependentAttentionModel,
        "model.",
        independent_model_paths,
        train_trajectories["image_trajectories"][training_index][0],
        train_trajectories["missions"][training_index],
    )
    independent_model_valid_evaluations = predict_evaluations(
        babyaiutil.models.discriminator.independent_attention.IndependentAttentionModel,
        "model.",
        independent_model_paths,
        valid_trajectories["image_trajectories"][validation_index][0],
        valid_trajectories["missions"][validation_index],
    )

    simple_attention_model_training_evaluations = predict_evaluations(
        babyaiutil.models.discriminator.simple_attention.SimpleAttentionModel,
        "model.",
        simple_attention_model_paths,
        train_trajectories["image_trajectories"][training_index][0],
        train_trajectories["missions"][training_index],
    )
    simple_attention_model_valid_evaluations = predict_evaluations(
        babyaiutil.models.discriminator.simple_attention.SimpleAttentionModel,
        "model.",
        simple_attention_model_paths,
        valid_trajectories["image_trajectories"][validation_index][0],
        valid_trajectories["missions"][validation_index],
    )

    fig = plot_qualitative(
        render_integer_encoded_grid(
            train_trajectories["image_trajectories"][training_index][0],
            train_trajectories["direction_trajectories"][training_index][0],
            64,
        ),
        " ".join([words[w] for w in train_trajectories["missions"][training_index]]),
        render_integer_encoded_grid(
            valid_trajectories["image_trajectories"][validation_index][0],
            valid_trajectories["direction_trajectories"][validation_index][0],
            64,
        ),
        " ".join([words[w] for w in valid_trajectories["missions"][validation_index]]),
        film_model_training_evaluations.mean(axis=0),
        film_model_valid_evaluations.mean(axis=0),
        film_model_valid_evaluations.std(axis=0),
        transformer_model_training_evaluations.mean(axis=0),
        transformer_model_valid_evaluations.mean(axis=0),
        transformer_model_valid_evaluations.std(axis=0),
        independent_noreg_model_training_evaluations.mean(axis=0),
        independent_noreg_model_valid_evaluations.mean(axis=0),
        independent_noreg_model_valid_evaluations.std(axis=0),
        simple_attention_model_training_evaluations.mean(axis=0),
        simple_attention_model_valid_evaluations.mean(axis=0),
        simple_attention_model_valid_evaluations.std(axis=0),
        independent_model_training_evaluations.mean(axis=0),
        independent_model_valid_evaluations.mean(axis=0),
        independent_model_valid_evaluations.std(axis=0),
        threshold=threshold,
    )

    if savefig:
        fig.savefig(savefig)


evaluate_and_plot_qualitative(
    0, 6, savefig="interaction_qualitative_evaluations.pdf", threshold=0.01
)
