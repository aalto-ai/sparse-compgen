import argparse
import os
import itertools
import json

import numpy as np
import torch
from tqdm.auto import tqdm

from common import (
    get_checkpoint_scores_tuples_for_models,
    trim_rows_indices_pcts,
    seed_limit_pairs_to_model_paths,
    flatten_tree,
    to_lists,
)


DEFAULT_LIMITS = [10, 50, 100, 250, 500, 1000, 2500, 5000, 9980]
DEFAULT_MODELS = ("independent", "independent_noreg", "simple")


def normalize(vec):
    return vec / np.linalg.norm(vec, axis=-1)[:, None]


def compute_normalized_outer_product(first, second):
    result = normalize(first) @ normalize(second).T
    return result


def fetch_numpy_array_from_sd(state_dict, key):
    return state_dict[key].cpu().numpy()


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
                lambda x: torch.load(x, map_location=torch.device("cpu")),
                model_paths,
            )
        ]
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
                lambda x: torch.load(x, map_location=torch.device("cpu")),
                model_paths,
            )
        ]
    )


CORRELATION_MAT_FUNCS = {
    "simple": get_projected_embedding_correlation_matrices_for_models
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs_dir")
    parser.add_argument("models_dir")
    parser.add_argument("output_json", type=str)
    parser.add_argument(
        "--experiment", type=str, default="discriminator_sample_nogoal_16_logsumexp"
    )
    parser.add_argument("--models", nargs="*", type=str, default=DEFAULT_MODELS)
    parser.add_argument("--limits", nargs="*", type=int, default=DEFAULT_LIMITS)
    args = parser.parse_args()

    scores = {
        m: get_checkpoint_scores_tuples_for_models(
            args.logs_dir,
            args.experiment,
            m,
            200000,
            1024,
            range(10),
            args.limits or DEFAULT_LIMITS,
        )
        for m in tqdm(args.models, desc="Computing scores")
    }
    row_indices = {
        m: trim_rows_indices_pcts(
            scores, limit_lowerbound=1, limit_upperbound=1000, pct=0.75
        )
        for m, scores in tqdm(scores.items(), desc="Computing indices")
    }
    model_paths = {
        m: seed_limit_pairs_to_model_paths(
            args.models_dir,
            args.experiment,
            m,
            200000,
            1024,
            flatten_tree(row_indices[m]),
        )
        for m in args.models
    }
    correlation_mats = {
        m: CORRELATION_MAT_FUNCS.get(m, get_embedding_correlation_matrices_for_models)(
            paths
        )
        for m, paths in tqdm(model_paths.items(), desc="Computing correlations")
    }

    with open(args.output_json, "w") as f:
        f.write(
            json.dumps(
                to_lists(
                    {
                        "means": {
                            k: v.mean(axis=0) for k, v in correlation_mats.items()
                        },
                        "std": {k: v.std(axis=0) for k, v in correlation_mats.items()},
                    }
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
